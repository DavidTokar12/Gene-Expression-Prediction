from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from gene_expression_prediction.data_loader import BigWigReader
from gene_expression_prediction.data_loader import GeneReader


class FeatureNames(Enum):
    """Histone marks and chromatin accessibility features."""

    DNASE = "dnase"
    H3K4ME1 = "h3k4me1"
    H3K4ME3 = "h3k4me3"
    H3K9ME3 = "h3k9me3"
    H3K27AC = "h3k27ac"
    H3K27ME3 = "h3k27me3"
    H3K36ME3 = "h3k36me3"


class CellLine(Enum):
    """Cell line identifiers."""

    X1 = "X1"
    X2 = "X2"
    X3 = "X3"


class SingleGeneData(TypedDict):
    """A dictionary holding all processed data for a single gene."""

    gene_info: pd.Series
    sequence_tensor_slice: np.ndarray
    target: float | None


@dataclass
class ProcessedFeatures:
    """
    Holds all processed data for a cell line using uniform binning.

    Following Enformer's approach:
    - Uniform 128bp bins across the entire window
    - Window centered on TSS (typically ±100kb)
    - Simple, consistent data structure

    Attributes:
        gene_annotations: Gene metadata (n_genes, n_annotation_columns)
        sequence_signal_tensor: Chromatin features (n_genes, n_bins, n_features)
        window_size: Total genomic window size in bp (e.g., 200000 for ±100kb)
        bin_size: Size of each bin in bp (typically 128)
        n_bins: Total number of bins
        target_expression: Gene expression targets (n_genes,)
    """

    gene_annotations: pd.DataFrame
    sequence_signal_tensor: np.ndarray
    window_size: int
    bin_size: int
    n_bins: int
    target_expression: pd.Series | None

    def get_gene_data_by_index(self, index: int) -> SingleGeneData:
        """Retrieves all data for a single gene at a specific integer index."""
        if not (0 <= index < len(self.gene_annotations)):
            raise IndexError(
                f"Index {index} is out of bounds (0-{len(self.gene_annotations) - 1})."
            )

        gene_name = self.gene_annotations.iloc[index]["gene_name"]

        return {
            "gene_info": self.gene_annotations.iloc[index],
            "sequence_tensor_slice": self.sequence_signal_tensor[index],
            "target": self.target_expression.loc[gene_name]
            if self.target_expression is not None
            else None,
        }

    def get_gene_data_by_name(self, gene_name: str) -> SingleGeneData:
        """Retrieves all data for a single gene by its unique name."""
        matches = self.gene_annotations.index[
            self.gene_annotations["gene_name"] == gene_name
        ].tolist()
        if not matches:
            raise KeyError(f"Gene '{gene_name}' not found in annotations.")

        index = matches[0]
        return self.get_gene_data_by_index(index)

    @property
    def half_window_size(self) -> int:
        """Returns the distance from TSS to window edge (±distance)."""
        return self.window_size // 2

def _create_uniform_sequence_tensor(
    gene_info_df: pd.DataFrame,
    bigwig_reader: BigWigReader,
    cell_line: CellLine,
    window_size: int,
    bin_size: int,
) -> tuple[np.ndarray, int]:
    """
    Creates uniform resolution sequence tensor following Enformer's approach.

    Uses consistent bin size across the entire genomic window centered on TSS.
    This captures both proximal promoter elements and distal enhancers with
    uniform resolution, making it ideal for transformer-based models.

    Args:
        gene_info_df: DataFrame with gene annotations including TSS positions
        bigwig_reader: BigWigReader instance with chromatin feature data
        cell_line: Which cell line to process
        window_size: Total genomic window size in bp (e.g., 200000 for ±100kb)
        bin_size: Size of each bin in bp (typically 128)

    Returns:
        Tuple of (tensor, n_bins)
        tensor shape: (n_genes, n_bins, n_features)

    Example (Corrected Calculation):
        With window_size=200000 and bin_size=128:
        - n_bins = 200000 // 128 = 1562
        - actual_coverage = 1562 * 128 = 199,936 bp
        - Window: ±99,968 bp from TSS (NOT ±100,000 bp)
        - Bins: 1562 bins of 128bp each
    """

    n_genes = len(gene_info_df)
    n_features = len(FeatureNames)

    n_bins = window_size // bin_size
    if n_bins == 0:
        raise ValueError(
            f"window_size ({window_size}) is smaller than bin_size ({bin_size}). "
            "This results in 0 bins."
        )

    actual_coverage = n_bins * bin_size
    actual_half_window = actual_coverage // 2

    # If actual_coverage is not even (e.g., odd n_bins), the "halves"
    # might be slightly different. We'll use this half_window as the
    # distance from TSS, ensuring the total is n_bins.
    # (e.g., 1562 * 128 = 199,936. half = 99,968. tss-99,968 to tss+99,968)

    print(f"\n{'=' * 70}")
    print("Uniform Binning Configuration (Enformer-style):")
    print(f"{'=' * 70}")
    print(f"  Target window size:   {window_size:,} bp")
    print(f"  Bin size:             {bin_size} bp")
    print(f"  Computed n_bins:      {n_bins:,} (window_size // bin_size)")
    print(f"  Actual coverage:      {actual_coverage:,} bp (n_bins * bin_size)")
    print(f"  Window center:        TSS ±{actual_half_window:,} bp")
    print(f"  Number of features:   {n_features}")
    print(f"  Number of genes:      {n_genes:,}")
    print(f"  Tensor shape:         ({n_genes:,}, {n_bins:,}, {n_features})")
    print(f"  Memory (est.):        {(n_genes * n_bins * n_features * 4 / 1e9):.2f} GB")
    print(f"{'=' * 70}\n")

    signal_tensor = np.zeros((n_genes, n_bins, n_features), dtype=np.float32)

    bw_handlers = {
        feature.value: getattr(
            bigwig_reader, f"{feature.value}_{cell_line.name.lower()}"
        )
        for feature in FeatureNames
    }


    genes_with_padding = set()
    genes_with_nans = set()

    for i, (_, gene) in enumerate(
        tqdm(
            gene_info_df.iterrows(), total=n_genes, desc=f"Processing {cell_line.value}"
        )
    ):
        
        chrom = gene["chr"]
        tss = int(gene["TSS_start"])
        gene_name = gene["gene_name"]


        target_start = tss - actual_half_window
        target_end = tss + actual_half_window

        is_padded_gene = False

        for j, feature in enumerate(FeatureNames):
            bw = bw_handlers[feature.value]

            chrom_len_str = bw.chroms().get(chrom)
            if not chrom_len_str:
                raise Exception(
                    f"Chromosome '{chrom}' (for gene '{gene_name}') "
                    f"not found in BigWig for feature '{feature.value}'"
                )
        
            chrom_len = int(chrom_len_str)

            query_start = max(0, target_start)
            query_end = min(chrom_len, target_end)

            # 4. If window is completely off-chromosome or invalid, skip
            # The tensor is already 0.0, which is correct (all padding)
            if query_start >= query_end:
                if not is_padded_gene:
                    genes_with_padding.add(gene_name)
                    is_padded_gene = True
                continue  # Go to next feature

            pad_bins_left = (query_start - target_start) // bin_size
            pad_bins_right = (target_end - query_end) // bin_size

            if pad_bins_left > 0 or (pad_bins_right > 0 and not is_padded_gene):
                    genes_with_padding.add(gene_name)
                    is_padded_gene = True

            query_n_bins = n_bins - pad_bins_left - pad_bins_right

            if query_n_bins <= 0:
                continue  # Go to next feature


            try:
                means = bw.stats(
                    chrom, query_start, query_end, type="mean", nBins=query_n_bins
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"pyBigWig query failed for gene '{gene_name}' (feature '{feature.value}') "
                    f"with query_start={query_start}, query_end={query_end}, nBins={query_n_bins}. "
                    f"Target was [{target_start}, {target_end}] on chrom {chrom} (len {chrom_len}). "
                    f"Original error: {e}"
                ) from e

            if means is None:
                # Valid region had no data, will be left as 0s
                continue

            valid_means_array = np.array([m if m is not None else 0.0 for m in means])

            final_means_array = np.pad(
                valid_means_array,
                (pad_bins_left, pad_bins_right),
                'constant',
                constant_values=0.0,
            )

            if final_means_array.shape[0] != n_bins:
                final_means_array.resize((n_bins,))

            signal_tensor[i, :, j] = final_means_array

            if np.any(np.isnan(valid_means_array)):
                genes_with_nans.add(gene_name)
                signal_tensor[i, :, j] = np.nan_to_num(
                    signal_tensor[i, :, j], nan=0.0
                )
    
    if genes_with_padding:
        print(
            f"\nInfo: {len(genes_with_padding)} genes were padded with zeros "
            "due to proximity to chromosome ends."
        )
        print(f"  (Example padded genes: {list(genes_with_padding)[:5]})")

    if genes_with_nans:
        print(
            f"\nWarning: {len(genes_with_nans)} genes had NaN values in BigWig "
            "(filled with 0)"
        )
        print(f"  (Example NaN genes: {list(genes_with_nans)[:5]})")

    return signal_tensor, n_bins


def process_cell_line(
    cell_line: CellLine,
    gene_reader: GeneReader,
    bigwig_reader: BigWigReader,
    window_size: int = 199_936,
    bin_size: int = 128,
    sample_n: int | None = None,
) -> ProcessedFeatures:
    """
    Process a single cell line with uniform binning (Enformer-style).

    Args:
        cell_line: Which cell line to process (X1, X2, or X3)
        gene_reader: GeneReader instance for loading gene annotations
        bigwig_reader: BigWigReader instance for loading chromatin features
        window_size: Total genomic window in bp (default: 200kb for ±100kb)
        bin_size: Size of each bin in bp (default: 128bp, matching Enformer)
        sample_n: Optional number of genes to sample for testing

    Returns:
        ProcessedFeatures object containing all processed data
    """

    if cell_line in [CellLine.X1, CellLine.X2]:
        train_info = gene_reader.load_gene_info(cell_line.value, "train")
        val_info = gene_reader.load_gene_info(cell_line.value, "val")
        gene_info_df = pd.concat([train_info, val_info], ignore_index=True)
    else:  # cell_line == CellLine.X3
        gene_info_df = gene_reader.load_gene_info(cell_line.value, "test")

    if sample_n is not None:
        print(f"Sampling {sample_n} genes from the original {len(gene_info_df)}.")
        gene_info_df = gene_info_df.sample(n=sample_n, random_state=42).reset_index(
            drop=True
        )

    # Create multiscale sequence tensor
    signal_tensor, n_bins = _create_uniform_sequence_tensor(
        gene_info_df,
        bigwig_reader,
        cell_line,
        window_size,
        bin_size,
    )

    # Load targets
    targets = None
    if cell_line in [CellLine.X1, CellLine.X2]:
        train_y = gene_reader.load_gene_expression(cell_line.value, "train")
        val_y = gene_reader.load_gene_expression(cell_line.value, "val")
        gex_df = pd.concat([train_y, val_y]).set_index("gene_name")
        targets = gex_df.reindex(gene_info_df["gene_name"])["gex"]

    return ProcessedFeatures(
        gene_annotations=gene_info_df,
        sequence_signal_tensor=signal_tensor,
        window_size=window_size,
        bin_size=bin_size,
        n_bins=n_bins,
        target_expression=targets,
    )


def save_processed_features(
    features: ProcessedFeatures, directory_path: str | Path
) -> None:
    """
    Saves ProcessedFeatures to disk.

    Args:
        features: The ProcessedFeatures object to save
        directory_path: Directory where files will be saved
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)

    features.gene_annotations.to_parquet(path / "gene_annotations.parquet")

    np.save(path / "sequence_signal_tensor.npy", features.sequence_signal_tensor)

    config = {
        "window_size": features.window_size,
        "bin_size": features.bin_size,
        "n_bins": features.n_bins,
    }
    np.save(path / "config.npy", config)

    if features.target_expression is not None:
        features.target_expression.to_csv(path / "target_expression.csv", header=True)

    print(f"✓ Saved processed features to: {path}")


def load_processed_features(directory_path: str | Path) -> ProcessedFeatures:
    """
    Loads ProcessedFeatures from disk.

    Args:
        directory_path: Directory containing the saved files

    Returns:
        Reconstructed ProcessedFeatures object
    """
    path = Path(directory_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    annotations = pd.read_parquet(path / "gene_annotations.parquet")
    signal_tensor = np.load(path / "sequence_signal_tensor.npy")
    config = np.load(path / "config.npy", allow_pickle=True).item()

    targets = None
    target_path = path / "target_expression.csv"
    if target_path.exists():
        targets = pd.read_csv(target_path, index_col=0, header=0).squeeze("columns")

    print(f"✓ Loaded processed features from: {path}")

    return ProcessedFeatures(
        gene_annotations=annotations,
        sequence_signal_tensor=signal_tensor,
        window_size=config["window_size"],
        bin_size=config["bin_size"],
        n_bins=config["n_bins"],
        target_expression=targets,
    )


def process_all_cell_lines(
    data_path: str,
    window_size: int = 199_936,
    bin_size: int = 128,
    sample_n: int | None = None,
    save_to_disk: bool = True,
    output_dir: str | None = None,
) -> tuple[ProcessedFeatures, ProcessedFeatures, ProcessedFeatures]:
    """
    Process all three cell lines (X1, X2, X3) with uniform binning.

    This is the main entry point for data processing. It follows Enformer's
    approach with uniform 128bp bins across a 200kb window (±100kb from TSS).

    Args:
        data_path: Path to data directory with gene info and BigWig files
        window_size: Total genomic window in bp (default: 200,000 = ±100kb)
        bin_size: Size of each bin in bp (default: 128bp)
        sample_n: Optional number of genes to sample for testing
        save_to_disk: Whether to save processed features
        output_dir: Output directory (defaults to data_path)

    Returns:
        Tuple of (cell_line_x1, cell_line_x2, cell_line_x3) ProcessedFeatures

    Example:
        >>> features_x1, features_x2, features_x3 = process_all_cell_lines(
        ...     data_path="/path/to/data",
        ...     window_size=199_936,  # ±100kb window
        ...     bin_size=128,         # Enformer standard
        ... )
    """

    print(f"\n{'=' * 70}")
    print("GENE EXPRESSION FEATURE PROCESSING - ENFORMER STYLE")
    print(f"{'=' * 70}")
    print(f"Data path:     {data_path}")
    print(f"Window size:   {window_size:,} bp (±{window_size // 2:,} bp from TSS)")
    print(f"Bin size:      {bin_size} bp")
    print(f"Total bins:    {window_size // bin_size}")
    print(f"Sample size:   {'All genes' if sample_n is None else f'{sample_n} genes'}")
    print(f"Save to disk:  {save_to_disk}")
    print(f"{'=' * 70}\n")

    print("Loading data readers...")
    gene_reader = GeneReader(data_path)
    bigwig_reader = BigWigReader(data_path)

    if output_dir is None:
        output_dir = data_path

    print("\n" + "=" * 70)
    print("PROCESSING CELL LINE X1 (Training)")
    print("=" * 70)
    cell_line_x1 = process_cell_line(
        cell_line=CellLine.X1,
        gene_reader=gene_reader,
        bigwig_reader=bigwig_reader,
        window_size=window_size,
        bin_size=bin_size,
        sample_n=sample_n,
    )

    if save_to_disk:
        output_path_x1 = Path(output_dir) / "processed_data_x1_enformer"
        save_processed_features(cell_line_x1, output_path_x1)

    print("\n" + "=" * 70)
    print("PROCESSING CELL LINE X2 (Validation)")
    print("=" * 70)
    cell_line_x2 = process_cell_line(
        cell_line=CellLine.X2,
        gene_reader=gene_reader,
        bigwig_reader=bigwig_reader,
        window_size=window_size,
        bin_size=bin_size,
        sample_n=sample_n,
    )

    if save_to_disk:
        output_path_x2 = Path(output_dir) / "processed_data_x2_enformer"
        save_processed_features(cell_line_x2, output_path_x2)

    print("\n" + "=" * 70)
    print("PROCESSING CELL LINE X3 (Test - using all genes)")
    print("=" * 70)
    cell_line_x3 = process_cell_line(
        cell_line=CellLine.X3,
        gene_reader=gene_reader,
        bigwig_reader=bigwig_reader,
        window_size=window_size,
        bin_size=bin_size,
        sample_n=None,
    )

    if save_to_disk:
        output_path_x3 = Path(output_dir) / "processed_data_x3_enformer"
        save_processed_features(cell_line_x3, output_path_x3)

    print("\n" + "=" * 70)
    print("✓ ALL CELL LINES PROCESSED SUCCESSFULLY!")
    print("=" * 70)

    return cell_line_x1, cell_line_x2, cell_line_x3
