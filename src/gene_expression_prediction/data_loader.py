from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyBigWig
import pyranges as pr


class GeneReader:
    def __init__(self, base_data_path: str | Path):
        """
        Initializes the DataLoader and validates the expected data paths.

        Args:
            base_data_path: The root directory containing the project data.

        Raises:
            FileNotFoundError: If any of the required files or directories are not found.
        """
        self.__base_path = Path(base_data_path)
        self._validate_paths()

        self._cage_train_x1_train_info: pd.DataFrame | None = None
        self._cage_train_x1_train_y: pd.DataFrame | None = None
        self._cage_train_x1_val_info: pd.DataFrame | None = None
        self._cage_train_x1_val_y: pd.DataFrame | None = None
        self._cage_train_x2_train_info: pd.DataFrame | None = None
        self._cage_train_x2_train_y: pd.DataFrame | None = None
        self._cage_train_x2_val_info: pd.DataFrame | None = None
        self._cage_train_x2_val_y: pd.DataFrame | None = None
        self._cage_train_x3_test_info: pd.DataFrame | None = None

    def _validate_paths(self):
        """Checks if all the required data files and directories exist."""
        cage_path = self.__base_path / "cage_train"
        required_files = [
            cage_path / "X1_train_info.tsv",
            cage_path / "X1_train_y.tsv",
            cage_path / "X1_val_info.tsv",
            cage_path / "X1_val_y.tsv",
            cage_path / "X2_train_info.tsv",
            cage_path / "X2_train_y.tsv",
            cage_path / "X2_val_info.tsv",
            cage_path / "X2_val_y.tsv",
            cage_path / "X3_test_info.tsv",
        ]

        if not cage_path.is_dir():
            raise FileNotFoundError(f"The directory '{cage_path}' was not found.")

        for file_path in required_files:
            if not file_path.is_file():
                raise FileNotFoundError(f"The data file '{file_path}' was not found.")

        print("All required data paths have been successfully validated.")

    def _load_tsv(self, file_path: Path) -> pd.DataFrame:
        """Helper function to load a tab-separated file."""
        return pd.read_csv(file_path, sep="\t")

    def load_gene_info(self, cell_line: str, split: str) -> pd.DataFrame:
        """
        Dynamically loads gene info data based on cell line and split.

        Args:
            cell_line: The cell line ('X1', 'X2', or 'X3').
            split: The data split ('train', 'val', or 'test').

        Returns:
            A pandas DataFrame with the requested gene information.
        """
        property_name = f"cage_train_{cell_line.lower()}_{split}_info"
        if hasattr(self, property_name):
            return getattr(self, property_name)
        else:
            raise ValueError(
                f"Invalid combination: cell_line='{cell_line}', split='{split}' for gene info."
            )

    def load_gene_expression(self, cell_line: str, split: str) -> pd.DataFrame:
        """
        Dynamically loads gene expression data based on cell line and split.

        Args:
            cell_line: The cell line ('X1' or 'X2').
            split: The data split ('train' or 'val').

        Returns:
            A pandas DataFrame with the requested gene expression data.
        """

        if cell_line == "X3":
            raise ValueError("No expression data available for the test set (X3).")

        property_name = f"cage_train_{cell_line.lower()}_{split}_y"
        if hasattr(self, property_name):
            return getattr(self, property_name)
        else:
            raise ValueError(
                f"Invalid combination: cell_line='{cell_line}', split='{split}' for gene expression."
            )

    @property
    def cage_train_x1_train_info(self) -> pd.DataFrame:
        if self._cage_train_x1_train_info is None:
            path = self.__base_path / "cage_train" / "X1_train_info.tsv"
            self._cage_train_x1_train_info = self._load_tsv(path)
        return self._cage_train_x1_train_info

    @property
    def cage_train_x1_train_y(self) -> pd.DataFrame:
        if self._cage_train_x1_train_y is None:
            path = self.__base_path / "cage_train" / "X1_train_y.tsv"
            self._cage_train_x1_train_y = self._load_tsv(path)
        return self._cage_train_x1_train_y

    @property
    def cage_train_x1_val_info(self) -> pd.DataFrame:
        if self._cage_train_x1_val_info is None:
            path = self.__base_path / "cage_train" / "X1_val_info.tsv"
            self._cage_train_x1_val_info = self._load_tsv(path)
        return self._cage_train_x1_val_info

    @property
    def cage_train_x1_val_y(self) -> pd.DataFrame:
        if self._cage_train_x1_val_y is None:
            path = self.__base_path / "cage_train" / "X1_val_y.tsv"
            self._cage_train_x1_val_y = self._load_tsv(path)
        return self._cage_train_x1_val_y

    @property
    def cage_train_x2_train_info(self) -> pd.DataFrame:
        if self._cage_train_x2_train_info is None:
            path = self.__base_path / "cage_train" / "X2_train_info.tsv"
            self._cage_train_x2_train_info = self._load_tsv(path)
        return self._cage_train_x2_train_info

    @property
    def cage_train_x2_train_y(self) -> pd.DataFrame:
        if self._cage_train_x2_train_y is None:
            path = self.__base_path / "cage_train" / "X2_train_y.tsv"
            self._cage_train_x2_train_y = self._load_tsv(path)
        return self._cage_train_x2_train_y

    @property
    def cage_train_x2_val_info(self) -> pd.DataFrame:
        if self._cage_train_x2_val_info is None:
            path = self.__base_path / "cage_train" / "X2_val_info.tsv"
            self._cage_train_x2_val_info = self._load_tsv(path)
        return self._cage_train_x2_val_info

    @property
    def cage_train_x2_val_y(self) -> pd.DataFrame:
        if self._cage_train_x2_val_y is None:
            path = self.__base_path / "cage_train" / "X2_val_y.tsv"
            self._cage_train_x2_val_y = self._load_tsv(path)
        return self._cage_train_x2_val_y

    @property
    def cage_train_x3_test_info(self) -> pd.DataFrame:
        if self._cage_train_x3_test_info is None:
            path = self.__base_path / "cage_train" / "X3_test_info.tsv"
            self._cage_train_x3_test_info = self._load_tsv(path)
        return self._cage_train_x3_test_info


class BedReader:
    """
    Reads BED peak data for the project's chromatin features.

    This class validates BED file paths on initialization and provides lazy-loading
    properties to access peak data. Data is loaded and cached upon first access.
    """

    def __init__(self, base_data_path: str | Path):
        """
        Initializes the BedReader, validates paths, and sets up cache attributes.
        """
        self.__base_path = Path(base_data_path)
        self.feature_names = [
            "dnase",
            "h3k4me1",
            "h3k4me3",
            "h3k9me3",
            "h3k27ac",
            "h3k27me3",
            "h3k36me3",
        ]
        self.cell_lines = ["X1", "X2", "X3"]
        self._validate_paths()

        self._dnase_x1: pr.PyRanges | None = None
        self._dnase_x2: pr.PyRanges | None = None
        self._dnase_x3: pr.PyRanges | None = None
        self._h3k4me1_x1: pr.PyRanges | None = None
        self._h3k4me1_x2: pr.PyRanges | None = None
        self._h3k4me1_x3: pr.PyRanges | None = None
        self._h3k4me3_x1: pr.PyRanges | None = None
        self._h3k4me3_x2: pr.PyRanges | None = None
        self._h3k4me3_x3: pr.PyRanges | None = None
        self._h3k9me3_x1: pr.PyRanges | None = None
        self._h3k9me3_x2: pr.PyRanges | None = None
        self._h3k9me3_x3: pr.PyRanges | None = None
        self._h3k27ac_x1: pr.PyRanges | None = None
        self._h3k27ac_x2: pr.PyRanges | None = None
        self._h3k27ac_x3: pr.PyRanges | None = None
        self._h3k27me3_x1: pr.PyRanges | None = None
        self._h3k27me3_x2: pr.PyRanges | None = None
        self._h3k27me3_x3: pr.PyRanges | None = None
        self._h3k36me3_x1: pr.PyRanges | None = None
        self._h3k36me3_x2: pr.PyRanges | None = None
        self._h3k36me3_x3: pr.PyRanges | None = None

    def _validate_paths(self):
        """Checks if all the required BED data files and directories exist."""
        bed_files_path = self.__base_path / "bed_files"
        if not bed_files_path.is_dir():
            raise FileNotFoundError(f"Directory not found: '{bed_files_path}'")

        for feature in self.feature_names:
            feature_dir = bed_files_path / f"{feature}_bed"
            if not feature_dir.is_dir():
                raise FileNotFoundError(f"Feature directory not found: '{feature_dir}'")
            for cell_line in self.cell_lines:
                bed_file = feature_dir / f"{cell_line}.bed"
                if not bed_file.is_file():
                    raise FileNotFoundError(f"BED file not found: '{bed_file}'")

    def _load_peaks(self, feature: str, cell_line: str) -> pr.PyRanges:
        path = self.__base_path / "bed_files" / f"{feature}_bed" / f"{cell_line}.bed"
        return pr.read_bed(str(path), as_df=False)

    @property
    def dnase_x1(self) -> pr.PyRanges:
        if self._dnase_x1 is None:
            self._dnase_x1 = self._load_peaks("dnase", "X1")
        return self._dnase_x1

    @property
    def dnase_x2(self) -> pr.PyRanges:
        if self._dnase_x2 is None:
            self._dnase_x2 = self._load_peaks("dnase", "X2")
        return self._dnase_x2

    @property
    def dnase_x3(self) -> pr.PyRanges:
        if self._dnase_x3 is None:
            self._dnase_x3 = self._load_peaks("dnase", "X3")
        return self._dnase_x3

    @property
    def h3k4me1_x1(self) -> pr.PyRanges:
        if self._h3k4me1_x1 is None:
            self._h3k4me1_x1 = self._load_peaks("h3k4me1", "X1")
        return self._h3k4me1_x1

    @property
    def h3k4me1_x2(self) -> pr.PyRanges:
        if self._h3k4me1_x2 is None:
            self._h3k4me1_x2 = self._load_peaks("h3k4me1", "X2")
        return self._h3k4me1_x2

    @property
    def h3k4me1_x3(self) -> pr.PyRanges:
        if self._h3k4me1_x3 is None:
            self._h3k4me1_x3 = self._load_peaks("h3k4me1", "X3")
        return self._h3k4me1_x3

    @property
    def h3k4me3_x1(self) -> pr.PyRanges:
        if self._h3k4me3_x1 is None:
            self._h3k4me3_x1 = self._load_peaks("h3k4me3", "X1")
        return self._h3k4me3_x1

    @property
    def h3k4me3_x2(self) -> pr.PyRanges:
        if self._h3k4me3_x2 is None:
            self._h3k4me3_x2 = self._load_peaks("h3k4me3", "X2")
        return self._h3k4me3_x2

    @property
    def h3k4me3_x3(self) -> pr.PyRanges:
        if self._h3k4me3_x3 is None:
            self._h3k4me3_x3 = self._load_peaks("h3k4me3", "X3")
        return self._h3k4me3_x3

    @property
    def h3k9me3_x1(self) -> pr.PyRanges:
        if self._h3k9me3_x1 is None:
            self._h3k9me3_x1 = self._load_peaks("h3k9me3", "X1")
        return self._h3k9me3_x1

    @property
    def h3k9me3_x2(self) -> pr.PyRanges:
        if self._h3k9me3_x2 is None:
            self._h3k9me3_x2 = self._load_peaks("h3k9me3", "X2")
        return self._h3k9me3_x2

    @property
    def h3k9me3_x3(self) -> pr.PyRanges:
        if self._h3k9me3_x3 is None:
            self._h3k9me3_x3 = self._load_peaks("h3k9me3", "X3")
        return self._h3k9me3_x3

    @property
    def h3k27ac_x1(self) -> pr.PyRanges:
        if self._h3k27ac_x1 is None:
            self._h3k27ac_x1 = self._load_peaks("h3k27ac", "X1")
        return self._h3k27ac_x1

    @property
    def h3k27ac_x2(self) -> pr.PyRanges:
        if self._h3k27ac_x2 is None:
            self._h3k27ac_x2 = self._load_peaks("h3k27ac", "X2")
        return self._h3k27ac_x2

    @property
    def h3k27ac_x3(self) -> pr.PyRanges:
        if self._h3k27ac_x3 is None:
            self._h3k27ac_x3 = self._load_peaks("h3k27ac", "X3")
        return self._h3k27ac_x3

    @property
    def h3k27me3_x1(self) -> pr.PyRanges:
        if self._h3k27me3_x1 is None:
            self._h3k27me3_x1 = self._load_peaks("h3k27me3", "X1")
        return self._h3k27me3_x1

    @property
    def h3k27me3_x2(self) -> pr.PyRanges:
        if self._h3k27me3_x2 is None:
            self._h3k27me3_x2 = self._load_peaks("h3k27me3", "X2")
        return self._h3k27me3_x2

    @property
    def h3k27me3_x3(self) -> pr.PyRanges:
        if self._h3k27me3_x3 is None:
            self._h3k27me3_x3 = self._load_peaks("h3k27me3", "X3")
        return self._h3k27me3_x3

    @property
    def h3k36me3_x1(self) -> pr.PyRanges:
        if self._h3k36me3_x1 is None:
            self._h3k36me3_x1 = self._load_peaks("h3k36me3", "X1")
        return self._h3k36me3_x1

    @property
    def h3k36me3_x2(self) -> pr.PyRanges:
        if self._h3k36me3_x2 is None:
            self._h3k36me3_x2 = self._load_peaks("h3k36me3", "X2")
        return self._h3k36me3_x2

    @property
    def h3k36me3_x3(self) -> pr.PyRanges:
        if self._h3k36me3_x3 is None:
            self._h3k36me3_x3 = self._load_peaks("h3k36me3", "X3")
        return self._h3k36me3_x3


class BigWigReader:
    """
    Reads BigWig continuous signal data for the project's chromatin features.

    This class validates BigWig file paths on initialization and provides
    lazy-loading properties to access the data. The BigWig files are opened
    and the file handlers are cached upon first access.
    """

    def __init__(self, base_data_path: str | Path):
        """
        Initializes the BigWigReader, validates paths, and sets up cache attributes.
        """
        self.__base_path = Path(base_data_path)
        self.feature_names = [
            "dnase",
            "h3k4me1",
            "h3k4me3",
            "h3k9me3",
            "h3k27ac",
            "h3k27me3",
            "h3k36me3",
        ]
        self.cell_lines = ["X1", "X2", "X3"]
        self._validate_paths()

        # Initialize cache attributes for all file handlers
        self._dnase_x1: pyBigWig.pyBigWig | None = None
        self._dnase_x2: pyBigWig.pyBigWig | None = None
        self._dnase_x3: pyBigWig.pyBigWig | None = None
        self._h3k4me1_x1: pyBigWig.pyBigWig | None = None
        self._h3k4me1_x2: pyBigWig.pyBigWig | None = None
        self._h3k4me1_x3: pyBigWig.pyBigWig | None = None
        self._h3k4me3_x1: pyBigWig.pyBigWig | None = None
        self._h3k4me3_x2: pyBigWig.pyBigWig | None = None
        self._h3k4me3_x3: pyBigWig.pyBigWig | None = None
        self._h3k9me3_x1: pyBigWig.pyBigWig | None = None
        self._h3k9me3_x2: pyBigWig.pyBigWig | None = None
        self._h3k9me3_x3: pyBigWig.pyBigWig | None = None
        self._h3k27ac_x1: pyBigWig.pyBigWig | None = None
        self._h3k27ac_x2: pyBigWig.pyBigWig | None = None
        self._h3k27ac_x3: pyBigWig.pyBigWig | None = None
        self._h3k27me3_x1: pyBigWig.pyBigWig | None = None
        self._h3k27me3_x2: pyBigWig.pyBigWig | None = None
        self._h3k27me3_x3: pyBigWig.pyBigWig | None = None
        self._h3k36me3_x1: pyBigWig.pyBigWig | None = None
        self._h3k36me3_x2: pyBigWig.pyBigWig | None = None
        self._h3k36me3_x3: pyBigWig.pyBigWig | None = None

    def _validate_paths(self):
        """
        Checks if all required BigWig data files and directories exist,
        handling both .bw and .bigwig extensions.
        """
        bw_files_path = self.__base_path / "bigwig_files"
        if not bw_files_path.is_dir():
            raise FileNotFoundError(f"Directory not found: '{bw_files_path}'")

        for feature in self.feature_names:
            feature_dir = bw_files_path / f"{feature}_bigwig"
            if not feature_dir.is_dir():
                raise FileNotFoundError(f"Feature directory not found: '{feature_dir}'")
            for cell_line in self.cell_lines:
                path_bw = feature_dir / f"{cell_line}.bw"
                path_bigwig = feature_dir / f"{cell_line}.bigwig"
                if not path_bw.is_file() and not path_bigwig.is_file():
                    raise FileNotFoundError(
                        f"BigWig file not found for {feature} {cell_line}. "
                        f"Checked for '{path_bw}' and '{path_bigwig}'."
                    )

    def _load_bigwig(self, feature: str, cell_line: str) -> pyBigWig.pyBigWig:
        """
        Loads a BigWig file, automatically handling .bw or .bigwig extensions.
        """
        feature_dir = self.__base_path / "bigwig_files" / f"{feature}_bigwig"
        path_bw = feature_dir / f"{cell_line}.bw"
        path_bigwig = feature_dir / f"{cell_line}.bigwig"

        final_path = path_bw if path_bw.is_file() else path_bigwig

        return pyBigWig.open(str(final_path))

    @property
    def dnase_x1(self) -> pyBigWig.pyBigWig:
        if self._dnase_x1 is None:
            self._dnase_x1 = self._load_bigwig("dnase", "X1")
        return self._dnase_x1

    @property
    def dnase_x2(self) -> pyBigWig.pyBigWig:
        if self._dnase_x2 is None:
            self._dnase_x2 = self._load_bigwig("dnase", "X2")
        return self._dnase_x2

    @property
    def dnase_x3(self) -> pyBigWig.pyBigWig:
        if self._dnase_x3 is None:
            self._dnase_x3 = self._load_bigwig("dnase", "X3")
        return self._dnase_x3

    @property
    def h3k4me1_x1(self) -> pyBigWig.pyBigWig:
        if self._h3k4me1_x1 is None:
            self._h3k4me1_x1 = self._load_bigwig("h3k4me1", "X1")
        return self._h3k4me1_x1

    @property
    def h3k4me1_x2(self) -> pyBigWig.pyBigWig:
        if self._h3k4me1_x2 is None:
            self._h3k4me1_x2 = self._load_bigwig("h3k4me1", "X2")
        return self._h3k4me1_x2

    @property
    def h3k4me1_x3(self) -> pyBigWig.pyBigWig:
        if self._h3k4me1_x3 is None:
            self._h3k4me1_x3 = self._load_bigwig("h3k4me1", "X3")
        return self._h3k4me1_x3

    @property
    def h3k4me3_x1(self) -> pyBigWig.pyBigWig:
        if self._h3k4me3_x1 is None:
            self._h3k4me3_x1 = self._load_bigwig("h3k4me3", "X1")
        return self._h3k4me3_x1

    @property
    def h3k4me3_x2(self) -> pyBigWig.pyBigWig:
        if self._h3k4me3_x2 is None:
            self._h3k4me3_x2 = self._load_bigwig("h3k4me3", "X2")
        return self._h3k4me3_x2

    @property
    def h3k4me3_x3(self) -> pyBigWig.pyBigWig:
        if self._h3k4me3_x3 is None:
            self._h3k4me3_x3 = self._load_bigwig("h3k4me3", "X3")
        return self._h3k4me3_x3

    @property
    def h3k9me3_x1(self) -> pyBigWig.pyBigWig:
        if self._h3k9me3_x1 is None:
            self._h3k9me3_x1 = self._load_bigwig("h3k9me3", "X1")
        return self._h3k9me3_x1

    @property
    def h3k9me3_x2(self) -> pyBigWig.pyBigWig:
        if self._h3k9me3_x2 is None:
            self._h3k9me3_x2 = self._load_bigwig("h3k9me3", "X2")
        return self._h3k9me3_x2

    @property
    def h3k9me3_x3(self) -> pyBigWig.pyBigWig:
        if self._h3k9me3_x3 is None:
            self._h3k9me3_x3 = self._load_bigwig("h3k9me3", "X3")
        return self._h3k9me3_x3

    @property
    def h3k27ac_x1(self) -> pyBigWig.pyBigWig:
        if self._h3k27ac_x1 is None:
            self._h3k27ac_x1 = self._load_bigwig("h3k27ac", "X1")
        return self._h3k27ac_x1

    @property
    def h3k27ac_x2(self) -> pyBigWig.pyBigWig:
        if self._h3k27ac_x2 is None:
            self._h3k27ac_x2 = self._load_bigwig("h3k27ac", "X2")
        return self._h3k27ac_x2

    @property
    def h3k27ac_x3(self) -> pyBigWig.pyBigWig:
        if self._h3k27ac_x3 is None:
            self._h3k27ac_x3 = self._load_bigwig("h3k27ac", "X3")
        return self._h3k27ac_x3

    @property
    def h3k27me3_x1(self) -> pyBigWig.pyBigWig:
        if self._h3k27me3_x1 is None:
            self._h3k27me3_x1 = self._load_bigwig("h3k27me3", "X1")
        return self._h3k27me3_x1

    @property
    def h3k27me3_x2(self) -> pyBigWig.pyBigWig:
        if self._h3k27me3_x2 is None:
            self._h3k27me3_x2 = self._load_bigwig("h3k27me3", "X2")
        return self._h3k27me3_x2

    @property
    def h3k27me3_x3(self) -> pyBigWig.pyBigWig:
        if self._h3k27me3_x3 is None:
            self._h3k27me3_x3 = self._load_bigwig("h3k27me3", "X3")
        return self._h3k27me3_x3

    @property
    def h3k36me3_x1(self) -> pyBigWig.pyBigWig:
        if self._h3k36me3_x1 is None:
            self._h3k36me3_x1 = self._load_bigwig("h3k36me3", "X1")
        return self._h3k36me3_x1

    @property
    def h3k36me3_x2(self) -> pyBigWig.pyBigWig:
        if self._h3k36me3_x2 is None:
            self._h3k36me3_x2 = self._load_bigwig("h3k36me3", "X2")
        return self._h3k36me3_x2

    @property
    def h3k36me3_x3(self) -> pyBigWig.pyBigWig:
        if self._h3k36me3_x3 is None:
            self._h3k36me3_x3 = self._load_bigwig("h3k36me3", "X3")
        return self._h3k36me3_x3


gene_info_df = GeneReader("/workspaces/Gene-Expression-Prediction/data").cage_train_x1_train_info


print(gene_info_df.head())

# print("PLUS STRAND GENES:")
# print(gene_info_df[gene_info_df['strand'] == '+'].head(3)[
#     ['chr', 'start', 'end', 'TSS_start', 'TSS_end', 'strand', 'gene_name']
# ])

# print("\nMINUS STRAND GENES:")
# print(gene_info_df[gene_info_df['strand'] == '-'].head(3)[
#     ['chr', 'start', 'end', 'TSS_start', 'TSS_end', 'strand', 'gene_name']
# ])