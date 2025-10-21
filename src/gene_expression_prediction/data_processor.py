from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr

from tqdm.notebook import tqdm

from gene_expression_prediction.data_loader import BedReader
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

    # A pandas Series containing the metadata for this gene (name, chr, TSS, etc.).
    gene_info: pd.Series
    # A numpy array slice for this gene from the main promoter tensor.
    # Shape: (n_bins, n_features, 3)
    promoter_tensor_slice: np.ndarray
    # A pandas Series containing all distal peak features for this gene.
    distal_features_row: pd.Series
    # The single floating-point gene expression value for this gene.
    target: float | None


@dataclass
class ProcessedFeatures:
    """
    A dataclass to hold all processed data for a cell line.

    This object contains the bulk data needed for model training and provides
    convenient methods to access all data for a single gene.
    """

    # Shape: (n_genes, n_annotation_columns)
    # Holds the original gene metadata (name, chr, TSS, etc.).
    gene_annotations: pd.DataFrame

    # Shape: (n_genes, n_bins, n_features, 3)
    # The main input for the CNN. Holds the binned signal data for the promoter
    # region. The 3 channels represent [mean, max, std] of the signal in each bin.
    promoter_signal_tensor: np.ndarray

    # Shape: (n_genes, n_distal_features)
    # The secondary input for the model. Holds the engineered features
    # (peak counts, max signal) from the distal enhancer regions.
    distal_peak_features: pd.DataFrame

    # Shape: (n_genes,)
    # A pandas Series containing the target gene expression values, indexed by gene_name.
    # Is None for the test set (cell line X3).
    target_expression: pd.Series | None

    def get_gene_data_by_index(self, index: int) -> SingleGeneData:
        """
        Retrieves all data for a single gene at a specific integer index.
        """
        if not (0 <= index < len(self.gene_annotations)):
            raise IndexError("Index is out of bounds.")

        gene_name = self.gene_annotations.iloc[index]["gene_name"]

        return {
            "gene_info": self.gene_annotations.iloc[index],
            "promoter_tensor_slice": self.promoter_signal_tensor[index],
            "distal_features_row": self.distal_peak_features.loc[gene_name],
            "target": self.target_expression.loc[gene_name]
            if self.target_expression is not None
            else None,
        }

    def get_gene_data_by_name(self, gene_name: str) -> SingleGeneData:
        """
        Retrieves all data for a single gene by its unique name.
        """
        matches = self.gene_annotations.index[
            self.gene_annotations["gene_name"] == gene_name
        ].tolist()
        if not matches:
            raise KeyError(f"Gene '{gene_name}' not found in annotations.")

        index = matches[0]
        return self.get_gene_data_by_index(index)


def _create_distal_bed_features(
    gene_info_df: pd.DataFrame,
    bed_reader: BedReader,
    cell_line: CellLine,
    distal_window_size: int,
    promoter_window_size: int,
) -> pd.DataFrame:
    """Generates BED-based features for the distal enhancer regions."""

    genes_pr = pr.PyRanges(
        gene_info_df.rename(
            columns={
                "chr": "Chromosome",
                "TSS_start": "Start",
                "TSS_end": "End",
                "strand": "Strand",
                "gene_name": "gene_id",  # Use a temporary, unique name
            }
        )
    )

    full_window = genes_pr.extend(distal_window_size)
    promoter_window_pr = genes_pr.extend(promoter_window_size)
    distal_regions = full_window.subtract(promoter_window_pr)
    master_df = gene_info_df[["gene_name"]].copy()

    for feature in tqdm(FeatureNames, desc="Distal BED Features"):
        peaks_pr = getattr(bed_reader, f"{feature.value}_{cell_line.name.lower()}")
        joined_pr = distal_regions.join(peaks_pr, strandedness=False)

        if joined_pr.empty:
            master_df[f"distal_{feature.value}_peak_count"] = 0
            master_df[f"distal_{feature.value}_max_signal"] = 0.0
            continue

        joined_df = joined_pr.df

        signal_col = "signalValue" if "signalValue" in joined_df.columns else "Score"

        agg_features = (
            joined_df.groupby("gene_id")
            .agg(
                peak_count=("Chromosome", "size"),
                max_signal=(signal_col, "max"),
            )
            .reset_index()
            .rename(
                columns={
                    "gene_id": "gene_name",  # Rename back to gene_name for the merge
                    "peak_count": f"distal_{feature.value}_peak_count",
                    "max_signal": f"distal_{feature.value}_max_signal",
                }
            )
        )

        master_df = pd.merge(master_df, agg_features, on="gene_name", how="left")
        master_df[f"distal_{feature.value}_peak_count"] = master_df[
            f"distal_{feature.value}_peak_count"
        ].fillna(0)
        master_df[f"distal_{feature.value}_max_signal"] = master_df[
            f"distal_{feature.value}_max_signal"
        ].fillna(0.0)

    return master_df.set_index("gene_name")


def _create_promoter_signal_tensor(
    gene_info_df: pd.DataFrame,
    bigwig_reader: BigWigReader,
    cell_line: CellLine,
    promoter_window_size: int,
    bin_size: int,
) -> np.ndarray:
    """Generates the 3-channel, strand-normalized tensor for the CNN."""

    n_genes = len(gene_info_df)
    n_features = len(FeatureNames)
    n_bins = (2 * promoter_window_size) // bin_size
    signal_tensor = np.zeros((n_genes, n_bins, n_features, 3), dtype=np.float32)

    bw_handlers = {
        feature.value: getattr(
            bigwig_reader, f"{feature.value}_{cell_line.name.lower()}"
        )
        for feature in FeatureNames
    }

    for i, gene in tqdm(
        gene_info_df.iterrows(), total=n_genes, desc="Promoter Signal Tensor"
    ):
        chrom = gene["chr"]
        strand = gene["strand"]
        tss = gene["TSS_start"]

        window_start = max(0, tss - promoter_window_size)
        window_end = tss + promoter_window_size

        for j, feature in enumerate(FeatureNames):
            bw = bw_handlers[feature.value]

            means = bw.stats(chrom, window_start, window_end, type="mean", nBins=n_bins)
            maxs = bw.stats(chrom, window_start, window_end, type="max", nBins=n_bins)
            min = bw.stats(chrom, window_start, window_end, type="min", nBins=n_bins)

            if strand == "-":
                means = means[::-1] if means is not None else None
                maxs = maxs[::-1] if maxs is not None else None
                min = min[::-1] if min is not None else None

            signal_tensor[i, :, j, 0] = np.nan_to_num(means, nan=0.0)
            signal_tensor[i, :, j, 1] = np.nan_to_num(maxs, nan=0.0)
            signal_tensor[i, :, j, 2] = np.nan_to_num(min, nan=0.0)

    return signal_tensor


def process_cell_line(
    cell_line: CellLine,
    gene_reader: GeneReader,
    bed_reader: BedReader,
    bigwig_reader: BigWigReader,
    promoter_window_size: int,
    bin_size: int,
    distal_window_size: int,
    sample_n: int | None = None,
) -> ProcessedFeatures:
    """
    Main entry point to generate all features for a given cell line.

    Args:
        cell_line (CellLine): The cell line to process (e.g., CellLine.X1).
        gene_reader (GeneReader): An instantiated reader for gene annotation files.
        bed_reader (BedReader): An instantiated reader for BED peak files.
        bigwig_reader (BigWigReader): An instantiated reader for BigWig signal files.
        promoter_window_size (int): The distance (+/-) from the TSS to define the promoter region for the CNN.
        bin_size (int): The size of each bin (in base pairs) for the promoter signal tensor.
        distal_window_size (int): The distance (+/-) from the TSS to define the distal region for BED features.
        sample_n (int | None, optional): If set, randomly sample this many genes to process. Defaults to None.
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

    distal_features_df = _create_distal_bed_features(
        gene_info_df, bed_reader, cell_line, distal_window_size, promoter_window_size
    )
    signal_tensor = _create_promoter_signal_tensor(
        gene_info_df, bigwig_reader, cell_line, promoter_window_size, bin_size
    )

    targets = None
    if cell_line in [CellLine.X1, CellLine.X2]:
        train_y = gene_reader.load_gene_expression(cell_line.value, "train")
        val_y = gene_reader.load_gene_expression(cell_line.value, "val")
        gex_df = pd.concat([train_y, val_y]).set_index("gene_name")
        targets = gex_df.reindex(gene_info_df["gene_name"])["gex"]

    return ProcessedFeatures(
        gene_annotations=gene_info_df,
        promoter_signal_tensor=signal_tensor,
        distal_peak_features=distal_features_df.reindex(gene_info_df["gene_name"]),
        target_expression=targets,
    )


def save_processed_features(features: ProcessedFeatures, directory_path: str | Path):
    """
    Saves the components of a ProcessedFeatures object to a directory.

    This uses efficient, standard formats: Parquet for DataFrames and .npy for NumPy arrays.

    Args:
        features: The ProcessedFeatures object to save.
        directory_path: The path to the directory where the files will be saved.
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)

    features.gene_annotations.to_parquet(path / "gene_annotations.parquet")
    features.distal_peak_features.to_parquet(path / "distal_peak_features.parquet")

    np.save(path / "promoter_signal_tensor.npy", features.promoter_signal_tensor)

    if features.target_expression is not None:
        features.target_expression.to_csv(path / "target_expression.csv", header=True)


def load_processed_features(directory_path: str | Path) -> ProcessedFeatures:
    """
    Loads processed features from a directory into a ProcessedFeatures object.

    Args:
        directory_path: The path to the directory containing the saved files.

    Returns:
        A reconstructed ProcessedFeatures object.
    """

    path = Path(directory_path)
    if not path.is_dir():
        raise FileNotFoundError(f"The directory '{path}' does not exist.")

    annotations = pd.read_parquet(path / "gene_annotations.parquet")
    distal_features = pd.read_parquet(path / "distal_peak_features.parquet")
    signal_tensor = np.load(path / "promoter_signal_tensor.npy")

    targets = None
    target_path = path / "target_expression.csv"
    if target_path.exists():
        targets = pd.read_csv(target_path, index_col=0, header=0).squeeze("columns")

    return ProcessedFeatures(
        gene_annotations=annotations,
        promoter_signal_tensor=signal_tensor,
        distal_peak_features=distal_features,
        target_expression=targets,
    )


def visualize_gene_data(
    gene_data: SingleGeneData, promoter_window_size: int, bin_size: int
):
    """
    Creates a comprehensive visualization for all processed data of a single gene.

    Args:
        gene_data: A SingleGeneData object containing all data for one gene.
        promoter_window_size: The +/- window size used for the promoter tensor.
        bin_size: The bin size used for the promoter tensor.
    """
    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(4, 3, height_ratios=[0.5, 2, 2, 4], hspace=0.6, wspace=0.3)

    ax_info = fig.add_subplot(gs[0, :])
    _plot_gene_info(ax_info, gene_data["gene_info"])

    ax_distal_count = fig.add_subplot(gs[1, :])
    ax_distal_signal = fig.add_subplot(gs[2, :])
    _plot_distal_features(
        ax_distal_count, ax_distal_signal, gene_data["distal_features_row"]
    )

    ax_mean = fig.add_subplot(gs[3, 0])
    ax_max = fig.add_subplot(gs[3, 1])
    ax_std = fig.add_subplot(gs[3, 2])
    _plot_promoter_tensor(
        [ax_mean, ax_max, ax_std],
        gene_data["promoter_tensor_slice"],
        promoter_window_size,
        bin_size,
    )

    plt.show()


def _plot_gene_info(ax: plt.Axes, gene_info: pd.Series):
    """Helper to display gene metadata as text."""
    ax.set_facecolor("#f0f0f0")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    info_text = ""
    for idx, val in gene_info.items():
        info_text += f"  {idx!s:<12}|  {val!s}\n"

    ax.text(
        0.01,
        0.95,
        info_text,
        va="top",
        ha="left",
        fontsize=12,
        fontfamily="monospace",
        wrap=True,
    )
    ax.set_title("Gene Annotations", fontsize=16, weight="bold")


def _plot_distal_features(
    ax_count: plt.Axes, ax_signal: plt.Axes, distal_features: pd.Series
):
    """Helper to plot distal features as two bar charts."""
    feature_names = [f.value for f in FeatureNames]

    counts = [
        distal_features.get(f"distal_{name}_peak_count", 0) for name in feature_names
    ]
    signals = [
        distal_features.get(f"distal_{name}_max_signal", 0) for name in feature_names
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    ax_count.bar(feature_names, counts, color=colors)
    ax_count.set_title(
        "Distal Peak Counts (-30kb to -5kb & +5kb to +30kb)", fontsize=16, weight="bold"
    )
    ax_count.set_ylabel("Number of Peaks")
    ax_count.tick_params(axis="x", rotation=45, labelsize=10)
    ax_count.grid(axis="y", linestyle="--", alpha=0.7)

    ax_signal.bar(feature_names, signals, color=colors)
    ax_signal.set_title(
        "Distal Max Signal (-30kb to -5kb & +5kb to +30kb)", fontsize=16, weight="bold"
    )
    ax_signal.set_ylabel("Max Signal Value")
    ax_signal.tick_params(axis="x", rotation=45, labelsize=10)
    ax_signal.grid(axis="y", linestyle="--", alpha=0.7)


def _plot_promoter_tensor(
    axes: list[plt.Axes],
    tensor_slice: np.ndarray,
    promoter_window_size: int,
    bin_size: int,
):
    """Helper to plot the 3 channels of the promoter tensor as heatmaps."""
    titles = ["Mean Signal", "Max Signal", "Min Signal"]
    n_bins, n_features, _ = tensor_slice.shape

    feature_labels = [f.value for f in FeatureNames]

    for i, ax in enumerate(axes):
        data_to_plot = tensor_slice[:, :, i].T

        im = ax.imshow(
            data_to_plot, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        ax.set_title(f"Promoter Tensor: {titles[i]}", fontsize=16, weight="bold")

        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(feature_labels)

        tick_positions = [0, n_bins // 2, n_bins - 1]
        tick_labels = [
            f"-{promoter_window_size / 1000:.0f}kb",
            "TSS",
            f"+{promoter_window_size / 1000:.0f}kb",
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Position relative to TSS")

        plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
