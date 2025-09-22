"""
Reptile Meta-Learning Training Script
Train Gemma models using Reptile algorithm for few-shot translation
"""

import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from reptile import ReptileMetaLearner, ReptileConfig
from clean_data import TEDDataCleaner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReptileTrainer:
    """Train Gemma models using Reptile meta-learning"""

    def __init__(
        self,
        config: ReptileConfig,
        data_dir: str = "datasets",
        token: Optional[str] = None,
        language_groups: Optional[List[str]] = None,
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.meta_learner = ReptileMetaLearner(config, token=token)
        self.language_groups = language_groups or []

    def load_training_data(self) -> List[Dict]:
        """Load and prepare training data from specified language groups."""
        all_tasks = []

        if not self.language_groups:
            raise ValueError("No language groups specified for training.")

        for group in self.language_groups:
            train_file = self.data_dir / f"{group}_train.pkl"

            if not train_file.exists():
                logger.info(
                    f"Training data file not found: {train_file}. Processing raw data..."
                )
                self._process_raw_data()

            if not train_file.exists():
                logger.warning(
                    f"Could not generate or find {train_file}. Skipping this group."
                )
                continue

            logger.info(f"Loading training data from {train_file}")
            with open(train_file, "rb") as f:
                df = pickle.load(f)
                # Convert DataFrame to list of dictionaries
                tasks = []
                for _, row in df.iterrows():
                    # Create tasks for all permutations of languages in the group, excluding self-translation
                    langs = [col for col in df.columns if col not in ["talk_name"]]
                    for src_lang in langs:
                        for tgt_lang in langs:
                            if src_lang != tgt_lang:
                                tasks.append(
                                    {
                                        "talk_name": row["talk_name"],
                                        "source_text": row[src_lang],
                                        "target_text": row[tgt_lang],
                                        "source_lang": src_lang,
                                        "target_lang": tgt_lang,
                                    }
                                )
                all_tasks.extend(tasks)

        if not all_tasks:
            raise FileNotFoundError(
                "No training data could be loaded. Please check the data files."
            )

        # The concept of 'task_type' needs to be created from the language columns
        for task in all_tasks:
            task["task_type"] = f"{task['source_lang']}_{task['target_lang']}"

        # Filter for base language pairs only
        base_task_types = set()
        if self.config.base_langs:
            for src in self.config.base_langs:
                for tgt in self.config.base_langs:
                    if src != tgt:
                        base_task_types.add(f"{src}_{tgt}")

        filtered_tasks = [
            task for task in all_tasks if task["task_type"] in base_task_types
        ]

        logger.info(
            f"Loaded {len(filtered_tasks)} training examples from groups: {', '.join(self.language_groups)}"
        )
        logger.info(f"Task types: {base_task_types}")

        return filtered_tasks

    def _process_raw_data(self):
        """Process raw TED data if processed data doesn't exist"""
        logger.info("Processing raw TED talk data...")
        cleaner = TEDDataCleaner(str(self.data_dir))
        cleaner.process_all_splits()
        logger.info("Data processing completed")

    def train(self, output_dir: str = "results") -> Dict:
        """Run Reptile meta-learning training"""
        logger.info("Starting Reptile meta-learning training")

        # Load training data
        train_tasks = self.load_training_data()

        if not train_tasks:
            raise ValueError("No training data available")

        # Run meta-learning
        training_history = self.meta_learner.train_meta_learning(train_tasks)

        # Prepare results
        results = {
            "config": {
                "meta_steps": self.config.meta_steps,
                "inner_steps": self.config.inner_steps,
                "support_size": self.config.support_size,
                "query_size": self.config.query_size,
                "model": self.config.gemma_model,
                "base_langs": self.config.base_langs,
                "target_langs": self.config.target_langs,
            },
            "training_history": training_history,
            "final_performance": {},
        }

        # Calculate final performance metrics
        for task_type, history in training_history.items():
            if history:
                results["final_performance"][task_type] = {
                    "initial": history[0] if len(history) > 0 else 0.0,
                    "final": history[-1] if len(history) > 0 else 0.0,
                    "improvement": (
                        (history[-1] - history[0]) if len(history) > 1 else 0.0
                    ),
                }

        # Save results
        self._save_training_results(results, output_dir)

        return results

    def _save_training_results(self, results: Dict, output_dir: str):
        """Save training results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = (
            output_path
            / f"reptile_training_{self.config.gemma_model.split('/')[-1]}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Training results saved to {results_file}")

        # Save training history as CSV for easy plotting
        self._save_training_csv(results["training_history"], output_path)

    def _save_training_csv(
        self, training_history: Dict[str, List[float]], output_path: Path
    ):
        """Save training history as CSV for analysis"""
        import pandas as pd

        # Convert to DataFrame
        max_length = max(
            len(history) for history in training_history.values() if history
        )

        csv_data = {}
        for task_type, history in training_history.items():
            # Pad shorter histories with the last value
            padded_history = (
                history + [history[-1]] * (max_length - len(history))
                if history
                else [0.0] * max_length
            )
            csv_data[task_type] = padded_history

        df = pd.DataFrame(csv_data)
        df.index.name = "meta_step"

        csv_file = (
            output_path
            / f"training_history_{self.config.gemma_model.split('/')[-1]}.csv"
        )
        df.to_csv(csv_file)

        logger.info(f"Training history CSV saved to {csv_file}")

    def print_training_summary(self, results: Dict):
        """Print training summary"""
        print("\n" + "=" * 60)
        print("REPTILE META-LEARNING TRAINING SUMMARY")
        print("=" * 60)

        config = results["config"]
        print(f"Model: {config['model']}")
        print(f"Meta-steps: {config['meta_steps']}")
        print(f"Inner steps: {config['inner_steps']}")
        print(f"Support size: {config['support_size']}")
        print(f"Base languages: {', '.join(config['base_langs'])}")

        print(f"\nFinal Performance:")
        for task_type, metrics in results["final_performance"].items():
            if task_type != "meta_average":
                print(
                    f"  {task_type}: {metrics['initial']:.3f} → {metrics['final']:.3f} "
                    f"(Δ{metrics['improvement']:+.3f})"
                )

        if "meta_average" in results["final_performance"]:
            meta_metrics = results["final_performance"]["meta_average"]
            print(
                f"\nMeta Average: {meta_metrics['initial']:.3f} → {meta_metrics['final']:.3f} "
                f"(Δ{meta_metrics['improvement']:+.3f})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Train Reptile meta-learning for translation"
    )
    parser.add_argument(
        "--model", choices=["270m", "1b"], default="270m", help="Model size to train"
    )
    parser.add_argument(
        "--meta_steps", type=int, default=100, help="Number of meta-learning steps"
    )
    parser.add_argument(
        "--inner_steps", type=int, default=5, help="Number of inner adaptation steps"
    )
    parser.add_argument(
        "--support_size",
        type=int,
        default=5,
        help="Number of support examples per episode",
    )
    parser.add_argument(
        "--query_size", type=int, default=3, help="Number of query examples per episode"
    )
    parser.add_argument("--data_dir", default="datasets", help="Data directory")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument(
        "--language_groups",
        nargs="+",
        default=["az_tr_en", "be_uk_en"],
        help="Language groups to use for training (e.g., az_tr_en be_uk_en)",
    )

    args = parser.parse_args()

    # Load .env file
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Map model choices to actual model names
    model_mapping = {"270m": "google/gemma-3-270m-it", "1b": "google/gemma-3-1b-it"}

    # Create configuration
    config = ReptileConfig(
        meta_steps=args.meta_steps,
        inner_steps=args.inner_steps,
        support_size=args.support_size,
        query_size=args.query_size,
        gemma_model=model_mapping[args.model],
    )

    # Initialize trainer
    trainer = ReptileTrainer(
        config, args.data_dir, token=token, language_groups=args.language_groups
    )

    try:
        # Run training
        results = trainer.train(args.output_dir)
        trainer.print_training_summary(results)

        logger.info("Reptile training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
