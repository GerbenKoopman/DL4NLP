"""
Simple TED Talk Data Cleaner
Removes __NULL__ entries and creates task dictionaries for meta-learning
"""

import pandas as pd
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TEDDataCleaner:
    """Clean and prepare TED talk data for meta-learning tasks"""

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        # Target languages for meta-learning
        self.base_langs = ["az", "be", "en"]  # Azerbaijani, Belarusian, English
        self.target_langs = ["tr", "uk"]  # Turkish, Ukrainian (for evaluation)

    def _get_available_langs(self, df: pd.DataFrame) -> List[str]:
        """Get available languages from the dataframe"""
        return [
            col for col in df.columns if col in (self.base_langs + self.target_langs)
        ]

    def clean_dataset(self, split_name: str) -> pd.DataFrame:
        """Clean a single dataset split based on language pairs."""
        file_path = self.data_dir / f"all_talks_{split_name}.tsv"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading {split_name} split from {file_path}")
        df = pd.read_csv(file_path, sep="\t")

        original_size = len(df)
        logger.info(f"Original size: {original_size} rows")

        # Define language pairs for cleaning
        lang_pairs = [("az", "tr"), ("be", "uk")]

        # Create a combined mask for rows to keep
        keep_mask = pd.Series(False, index=df.index)

        for lang1, lang2 in lang_pairs:
            if lang1 in df.columns and lang2 in df.columns:
                # Rows where both languages in a pair are not NULL
                pair_mask = ~df[lang1].astype(str).str.contains("NULL") & ~df[
                    lang2
                ].astype(str).str.contains("NULL")
                keep_mask |= pair_mask

        df = df[keep_mask]

        # Remove rows where 'en' is NULL
        if "en" in df.columns:
            df = df[~df["en"].astype(str).str.contains("NULL")]

        cleaned_size = len(df)
        logger.info(
            f"Cleaned size: {cleaned_size} rows ({original_size - cleaned_size} removed)"
        )

        # Dynamically select available columns from the cleaned data
        available_langs = self._get_available_langs(df)
        return df[["talk_name"] + available_langs]

    def save_language_groups(self, df: pd.DataFrame, split_name: str):
        """Save cleaned data into language-specific files."""
        lang_groups = {
            "az_tr_en": ["az", "tr", "en"],
            "be_uk_en": ["be", "uk", "en"],
        }

        for group_name, langs in lang_groups.items():
            # Ensure all required columns are present in the dataframe
            required_cols = ["talk_name"] + langs
            if not all(col in df.columns for col in langs):
                logger.warning(
                    f"Skipping group {group_name} for split {split_name}: Missing language columns"
                )
                continue

            # Select the relevant columns
            group_df = df[required_cols]

            # Drop rows where any of the group's languages are null
            # The 'en' column is already cleaned in the clean_dataset method
            pair_langs = [lang for lang in langs if lang != "en"]

            # Create a mask to identify rows with 'NULL' in any of the pair language columns.
            null_mask = (
                group_df[pair_langs]
                .apply(lambda x: x.astype(str).str.contains("NULL"))
                .any(axis=1)
            )

            # Invert the mask to keep rows that do not contain 'NULL'
            cleaned_group_df = group_df[~null_mask]

            if cleaned_group_df.empty:
                logger.info(
                    f"No data for group {group_name} in split {split_name} after cleaning."
                )
                continue

            # Save the cleaned dataframe to a pickle file
            output_path = self.data_dir / f"{group_name}_{split_name}.pkl"
            cleaned_group_df.to_pickle(output_path)
            logger.info(f"Saved {len(cleaned_group_df)} rows to {output_path}")

    def process_all_splits(self):
        """Process all dataset splits"""
        for split in ["train", "dev", "test"]:
            try:
                # Clean the dataset
                df = self.clean_dataset(split)

                # Save language-specific files
                self.save_language_groups(df, split)

            except FileNotFoundError as e:
                logger.warning(f"Skipping {split}: {e}")

        logger.info("Completed processing all splits")


def main():
    parser = argparse.ArgumentParser(description="Clean TED talk data")
    parser.add_argument("--data_dir", default="datasets", help="Data directory")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Which split to process",
    )

    args = parser.parse_args()

    cleaner = TEDDataCleaner(args.data_dir)

    if args.split == "all":
        cleaner.process_all_splits()
    else:
        df = cleaner.clean_dataset(args.split)
        cleaner.save_language_groups(df, args.split)


if __name__ == "__main__":
    main()
