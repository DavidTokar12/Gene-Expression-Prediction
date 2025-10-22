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
    DNASE = "dnase"
    H3K4ME1 = "h3k4me1"
    H3K4ME3 = "h3k4me3"
    H3K9ME3 = "h3k9me3"
    H3K27AC = "h3k27ac"
    H3K27ME3 = "h3k27me3"
    H3K36ME3 = "h3k36me3"


class CellLine(Enum):
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
    A dataclass to hold all processed data for a cell line.

    Uses multi-resolution binning: fine bins near TSS, coarse bins in distal regions.
    """

    # Shape: (n_genes, n_annotation_columns)
    gene_annotations: pd.DataFrame

    # Shape: (n_genes, n_bins_total, n_features, 2)
    # Multi-resolution sequence tensor with 2 channels: [mean, max]
    sequence_signal_tensor: np.ndarray

    # Number of bins in each region for reference
    n_upstream_bins: int
    n_promoter_bins: int
    n_downstream_bins: int
    n_total_bins: int

    # Shape: (n_genes,)
    target_expression: pd.Series | None

    def get_gene_data_by_index(self, index: int) -> SingleGeneData:
        """Retrieves all data for a single gene at a specific integer index."""
        if not (0 <= index < len(self.gene_annotations)):
            raise IndexError("Index is out of bounds.")

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


def _create_multiscale_sequence_tensor(
    gene_info_df: pd.DataFrame,
    bigwig_reader: BigWigReader,
    cell_line: CellLine,
    promoter_window_size: int,
    distal_window_size: int,
    promoter_bin_size: int,
    distal_bin_size: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Creates multi-resolution sequence tensor with finer bins near TSS.
    
    Structure:
    - Upstream distal: coarse bins (e.g., 5kb each)
    - Promoter: fine bins (e.g., 100bp each)
    - Downstream distal: coarse bins (e.g., 5kb each)
    
    Args:
        gene_info_df: DataFrame with gene annotations
        bigwig_reader: BigWigReader instance
        cell_line: Which cell line to process
        promoter_window_size: Distance from TSS for promoter region (e.g., 5000)
        distal_window_size: Distance from TSS for entire window (e.g., 50000)
        promoter_bin_size: Bin size for promoter region (e.g., 100)
        distal_bin_size: Bin size for distal regions (e.g., 5000)
    
    Returns:
        Tuple of (tensor, n_upstream_bins, n_promoter_bins, n_downstream_bins, n_total_bins)
        Tensor shape: (n_genes, n_total_bins, n_features, 3)
    """
    
    n_genes = len(gene_info_df)
    n_features = len(FeatureNames)
    
    # Calculate number of bins
    distal_span = distal_window_size - promoter_window_size
    n_distal_bins_per_side = distal_span // distal_bin_size
    n_promoter_bins = (2 * promoter_window_size) // promoter_bin_size
    n_total_bins = n_distal_bins_per_side * 2 + n_promoter_bins
    
    print(f"\n{'='*60}")
    print("Multi-resolution binning configuration:")
    print(f"{'='*60}")
    print(f"  Upstream distal:   {n_distal_bins_per_side} bins x {distal_bin_size}bp = {distal_span}bp")
    print(f"  Promoter:          {n_promoter_bins} bins × {promoter_bin_size}bp = {2*promoter_window_size}bp")
    print(f"  Downstream distal: {n_distal_bins_per_side} bins x {distal_bin_size}bp = {distal_span}bp")
    print(f"  Total bins:        {n_total_bins}")
    print(f"  Total span:        {2*distal_window_size}bp ({-distal_window_size} to +{distal_window_size})")
    print(f"{'='*60}\n")
    
    signal_tensor = np.zeros((n_genes, n_total_bins, n_features, 2), dtype=np.float32)
    
    bw_handlers = {
        feature.value: getattr(bigwig_reader, f"{feature.value}_{cell_line.name.lower()}")
        for feature in FeatureNames
    }
    
    nan_genes = []
    error_genes = []
    
    def safe_stats(bw, chrom, start, end, stat_type, n_bins):
        """Safely get stats with error handling."""
        try:
            # Ensure valid coordinates
            if start < 0:
                start = 0
            if start >= end:
                return None
            
            result = bw.stats(chrom, int(start), int(end), type=stat_type, nBins=int(n_bins))
            return result
        except Exception as e:
            return None
    
    for i, gene in tqdm(gene_info_df.iterrows(), total=n_genes, desc="Multiscale Sequence"):
        chrom = gene["chr"]
        strand = gene["strand"]
        tss = gene["TSS_start"]
        
        for j, feature in enumerate(FeatureNames):
            bw = bw_handlers[feature.value]
            
            if chrom not in bw.chroms():
                continue
            
            try:
                # Get signals for three regions separately
                
                # 1. Upstream distal region
                upstream_start = max(0, tss - distal_window_size)
                upstream_end = tss - promoter_window_size
                
                upstream_means = safe_stats(bw, chrom, upstream_start, upstream_end, "mean", n_distal_bins_per_side)
                upstream_maxs = safe_stats(bw, chrom, upstream_start, upstream_end, "max", n_distal_bins_per_side)
                
                # 2. Promoter region (high resolution)
                promoter_start = tss - promoter_window_size
                promoter_end = tss + promoter_window_size
                
                promoter_means = safe_stats(bw, chrom, promoter_start, promoter_end, "mean", n_promoter_bins)
                promoter_maxs = safe_stats(bw, chrom, promoter_start, promoter_end, "max", n_promoter_bins)
                
                # 3. Downstream distal region
                downstream_start = tss + promoter_window_size
                downstream_end = tss + distal_window_size
                
                downstream_means = safe_stats(bw, chrom, downstream_start, downstream_end, "mean", n_distal_bins_per_side)
                downstream_maxs = safe_stats(bw, chrom, downstream_start, downstream_end, "max", n_distal_bins_per_side)
                
                # Concatenate all regions into one sequence
                all_means = (list(upstream_means or [0.0] * n_distal_bins_per_side) + 
                            list(promoter_means or [0.0] * n_promoter_bins) + 
                            list(downstream_means or [0.0] * n_distal_bins_per_side))
                all_maxs = (list(upstream_maxs or [0.0] * n_distal_bins_per_side) + 
                           list(promoter_maxs or [0.0] * n_promoter_bins) + 
                           list(downstream_maxs or [0.0] * n_distal_bins_per_side))

                # Convert to arrays and handle None/NaN
                means_arr = np.array([0.0 if (s is None or np.isnan(s)) else s for s in all_means], dtype=np.float32)
                maxs_arr = np.array([0.0 if (s is None or np.isnan(s)) else s for s in all_maxs], dtype=np.float32)
                
                # Reverse entire sequence for negative strand
                if strand == "-":
                    means_arr = means_arr[::-1]
                    maxs_arr = maxs_arr[::-1]
                
                # Clean NaN/Inf
                means_arr = np.nan_to_num(means_arr, nan=0.0, posinf=0.0, neginf=0.0)
                maxs_arr = np.nan_to_num(maxs_arr, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Check for any remaining NaN
                if np.isnan(means_arr).any() or np.isnan(maxs_arr).any():
                    if gene["gene_name"] not in nan_genes:
                        nan_genes.append(gene["gene_name"])
                
                signal_tensor[i, :, j, 0] = means_arr
                signal_tensor[i, :, j, 1] = maxs_arr
                
            except Exception as e:
                if gene["gene_name"] not in error_genes:
                    error_genes.append(gene["gene_name"])
                    print(f"\nWarning: Error processing gene {gene['gene_name']}: {e!s}")
                continue
    
    # Final validation
    if error_genes:
        print(f"\n⚠️  WARNING: Encountered errors in {len(error_genes)} genes")
    
    if np.isnan(signal_tensor).any():
        print(f"\n⚠️  WARNING: Found NaN in {len(nan_genes)} genes: {nan_genes[:10]}")
        print("Replacing all NaN with 0.0")
        signal_tensor = np.nan_to_num(signal_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.isinf(signal_tensor).any():
        print(f"\n⚠️  WARNING: Found Inf values in tensor")
        print("Replacing all Inf with 0.0")
        signal_tensor = np.nan_to_num(signal_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n✅ Multiscale tensor created successfully:")
    print(f"   Shape: {signal_tensor.shape}")
    print(f"   Range: [{signal_tensor.min():.3f}, {signal_tensor.max():.3f}]")
    print(f"   NaN: {np.isnan(signal_tensor).any()}, Inf: {np.isinf(signal_tensor).any()}\n")
    
    return signal_tensor, n_distal_bins_per_side, n_promoter_bins, n_distal_bins_per_side, n_total_bins


def process_cell_line(
    cell_line: CellLine,
    gene_reader: GeneReader,
    bigwig_reader: BigWigReader,
    promoter_window_size: int,
    distal_window_size: int,
    promoter_bin_size: int,
    distal_bin_size: int,
    sample_n: int | None = None,
) -> ProcessedFeatures:
    """
    Main entry point to generate all features for a given cell line.

    Uses multi-resolution binning with only BigWig signal data.

    Args:
        cell_line: The cell line to process (e.g., CellLine.X1)
        gene_reader: An instantiated reader for gene annotation files
        bigwig_reader: An instantiated reader for BigWig signal files
        promoter_window_size: Distance (+/-) from TSS for promoter (e.g., 5000)
        distal_window_size: Distance (+/-) from TSS for entire window (e.g., 50000)
        promoter_bin_size: Bin size for promoter region in bp (e.g., 100)
        distal_bin_size: Bin size for distal regions in bp (e.g., 5000)
        sample_n: If set, randomly sample this many genes. Defaults to None.

    Returns:
        ProcessedFeatures object containing the multiscale sequence tensor
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
    signal_tensor, n_upstream, n_promoter, n_downstream, n_total = (
        _create_multiscale_sequence_tensor(
            gene_info_df,
            bigwig_reader,
            cell_line,
            promoter_window_size,
            distal_window_size,
            promoter_bin_size,
            distal_bin_size,
        )
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
        n_upstream_bins=n_upstream,
        n_promoter_bins=n_promoter,
        n_downstream_bins=n_downstream,
        n_total_bins=n_total,
        target_expression=targets,
    )


def save_processed_features(features: ProcessedFeatures, directory_path: str | Path):
    """
    Saves the components of a ProcessedFeatures object to a directory.

    Args:
        features: The ProcessedFeatures object to save
        directory_path: The path to the directory where files will be saved
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)

    features.gene_annotations.to_parquet(path / "gene_annotations.parquet")
    np.save(path / "sequence_signal_tensor.npy", features.sequence_signal_tensor)

    # Save bin configuration
    bin_config = {
        "n_upstream_bins": features.n_upstream_bins,
        "n_promoter_bins": features.n_promoter_bins,
        "n_downstream_bins": features.n_downstream_bins,
        "n_total_bins": features.n_total_bins,
    }
    np.save(path / "bin_config.npy", bin_config)

    if features.target_expression is not None:
        features.target_expression.to_csv(path / "target_expression.csv", header=True)


def load_processed_features(directory_path: str | Path) -> ProcessedFeatures:
    """
    Loads processed features from a directory into a ProcessedFeatures object.

    Args:
        directory_path: The path to the directory containing the saved files

    Returns:
        A reconstructed ProcessedFeatures object
    """
    path = Path(directory_path)
    if not path.is_dir():
        raise FileNotFoundError(f"The directory '{path}' does not exist.")

    annotations = pd.read_parquet(path / "gene_annotations.parquet")
    signal_tensor = np.load(path / "sequence_signal_tensor.npy")

    # Load bin configuration
    bin_config = np.load(path / "bin_config.npy", allow_pickle=True).item()

    targets = None
    target_path = path / "target_expression.csv"
    if target_path.exists():
        targets = pd.read_csv(target_path, index_col=0, header=0).squeeze("columns")

    return ProcessedFeatures(
        gene_annotations=annotations,
        sequence_signal_tensor=signal_tensor,
        n_upstream_bins=bin_config["n_upstream_bins"],
        n_promoter_bins=bin_config["n_promoter_bins"],
        n_downstream_bins=bin_config["n_downstream_bins"],
        n_total_bins=bin_config["n_total_bins"],
        target_expression=targets,
    )
