"""Centralized path management for the project."""

from pathlib import Path


class Paths:
    """A class to manage all file paths for the project."""

    def __init__(self, base_dir: Path | None = None):
        """
        Initializes the Paths object.

        Args:
            base_dir: The base directory of the project. If None, it is inferred
                by going up two levels from this file's location.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Assuming this file is in src/, so go up two levels
            self.base_dir = Path(__file__).resolve().parent.parent

        # Main directories
        self.src_dir = self.base_dir / "src"
        self.data_dir = self.base_dir / "datasets"
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.results_dir / "plots"

        # Data files
        self.all_talks_train = self.data_dir / "all_talks_train.tsv"
        self.all_talks_dev = self.data_dir / "all_talks_dev.tsv"
        self.all_talks_test = self.data_dir / "all_talks_test.tsv"

        # Pickled data files
        self.az_tr_en_train_pkl = self.data_dir / "az_tr_en_train.pkl"
        self.az_tr_en_dev_pkl = self.data_dir / "az_tr_en_dev.pkl"
        self.az_tr_en_test_pkl = self.data_dir / "az_tr_en_test.pkl"
        self.be_uk_en_train_pkl = self.data_dir / "be_uk_en_train.pkl"
        self.be_uk_en_dev_pkl = self.data_dir / "be_uk_en_dev.pkl"
        self.be_uk_en_test_pkl = self.data_dir / "be_uk_en_test.pkl"

        # Results files
        self.baseline_results = self.results_dir / "baseline_results.json"

    def get_results_dir(self, model_name: str) -> Path:
        """Returns the directory for storing results for a given model."""
        return self.results_dir / model_name

    def get_training_history_csv(self, model_name: str) -> Path:
        """Returns the path to the training history CSV for a given model."""
        return self.get_results_dir(model_name) / f"training_history_{model_name}.csv"

    def get_reptile_evaluation_summary_csv(self, model_name: str) -> Path:
        """Returns the path to the reptile evaluation summary CSV for a given model."""
        return (
            self.get_results_dir(model_name)
            / f"reptile_evaluation_summary_{model_name}.csv"
        )


# A global instance to be used throughout the project
paths = Paths()
