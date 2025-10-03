"""
Reptile Meta-Learning Evaluation Script
Evaluate meta-learned models on transfer tasks (Turkish, Ukrainian)
"""

import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

from reptile import ReptileMetaLearner, ReptileConfig
from clean_data import TEDDataCleaner
from paths import paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReptileEvaluator:
    """Evaluate Reptile meta-learned models on transfer tasks"""

    def __init__(
        self,
        config: ReptileConfig,
        token: Optional[str] = None,
        language_groups: Optional[List[str]] = None,
        wandb_api_key: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ):
        self.config = config
        self.meta_learner = ReptileMetaLearner(
            config,
            token=token,
            wandb_api_key=wandb_api_key,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
        self.language_groups = language_groups or []

    def load_test_data(self) -> List[Dict]:
        """Load and prepare test data from specified language groups."""
        all_tasks = []

        if not self.language_groups:
            raise ValueError("No language groups specified for evaluation.")

        for group in self.language_groups:
            test_file = paths.data_dir / f"{group}_test.pkl"

            if not test_file.exists():
                logger.info(
                    f"Test data file not found: {test_file}. Processing raw data..."
                )
                self._process_raw_data()

            if not test_file.exists():
                logger.warning(
                    f"Could not generate or find {test_file}. Skipping this group."
                )
                continue

            logger.info(f"Loading test data from {test_file}")
            with open(test_file, "rb") as f:
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
                "No test data could be loaded. Please check the data files."
            )

        # The concept of 'task_type' needs to be created from the language columns
        for task in all_tasks:
            task["task_type"] = f"{task['source_lang']}_{task['target_lang']}"

        logger.info(
            f"Loaded {len(all_tasks)} test examples from groups: {', '.join(self.language_groups)}"
        )
        return all_tasks

    def _process_raw_data(self):
        """Process raw TED data if processed data doesn't exist"""
        logger.info("Processing raw TED talk data...")
        cleaner = TEDDataCleaner()
        cleaner.process_all_splits()
        logger.info("Data processing completed")

    def evaluate_comprehensive(self, output_dir: str = "results") -> Dict:
        """Run comprehensive evaluation (zero-shot + few-shot + transfer)"""
        logger.info("Starting comprehensive Reptile evaluation")

        # Load test data
        test_tasks = self.load_test_data()

        if not test_tasks:
            raise ValueError("No test data available")

        results = {
            "config": {
                "model": self.config.gemma_model,
                "support_size": self.config.support_size,
                "base_langs": self.config.base_langs,
                "target_langs": self.config.target_langs,
            },
            "evaluations": {},
        }

        # 1. Zero-shot evaluation on all languages
        logger.info("Running zero-shot evaluation...")
        zero_shot_results = self.meta_learner.zero_shot_evaluate(test_tasks)
        results["evaluations"]["zero_shot"] = zero_shot_results

        # 2. Few-shot evaluation on base languages (1-shot and 5-shot)
        logger.info("Running few-shot evaluation on base languages...")
        base_tasks = [
            task
            for task in test_tasks
            if self.config.base_langs
            and any(lang in task["task_type"] for lang in self.config.base_langs)
        ]

        few_shot_1 = self.meta_learner.evaluate_transfer(base_tasks, num_shots=1)
        few_shot_5 = self.meta_learner.evaluate_transfer(base_tasks, num_shots=5)

        results["evaluations"]["few_shot_1"] = few_shot_1
        results["evaluations"]["few_shot_5"] = few_shot_5

        # 3. Transfer evaluation on target languages (Turkish, Ukrainian)
        logger.info("Running transfer evaluation on target languages...")
        target_tasks = [
            task
            for task in test_tasks
            if self.config.target_langs
            and any(lang in task["task_type"] for lang in self.config.target_langs)
        ]

        if target_tasks:
            transfer_1 = self.meta_learner.evaluate_transfer(target_tasks, num_shots=1)
            transfer_5 = self.meta_learner.evaluate_transfer(target_tasks, num_shots=5)

            results["evaluations"]["transfer_1"] = transfer_1
            results["evaluations"]["transfer_5"] = transfer_5
        else:
            logger.warning("No target language tasks found for transfer evaluation")
            results["evaluations"]["transfer_1"] = {}
            results["evaluations"]["transfer_5"] = {}

        # Calculate summary statistics
        self._calculate_summary_stats(results)

        # Save results
        self._save_evaluation_results(results, output_dir)

        return results

    def _calculate_summary_stats(self, results: Dict):
        """Calculate summary statistics across evaluations"""
        summary = {}

        for eval_type, eval_results in results["evaluations"].items():
            if eval_results:
                scores = list(eval_results.values())
                summary[eval_type] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }
            else:
                summary[eval_type] = {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        results["summary"] = summary

    def _save_evaluation_results(self, results: Dict, output_dir: str):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        model_name = self.config.gemma_model.split("/")[-1]
        results_file = output_path / f"reptile_evaluation_{model_name}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {results_file}")

        # Save summary CSV
        self._save_summary_csv(results, output_path, model_name)

    def _save_summary_csv(self, results: Dict, output_path: Path, model_name: str):
        """Save evaluation summary as CSV"""
        import pandas as pd

        # Prepare data for CSV
        csv_data = []

        for eval_type, eval_results in results["evaluations"].items():
            for task_type, score in eval_results.items():
                csv_data.append(
                    {
                        "model": model_name,
                        "evaluation_type": eval_type,
                        "task_type": task_type,
                        "score": score,
                    }
                )

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = output_path / f"reptile_evaluation_summary_{model_name}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Summary CSV saved to {csv_file}")

    def compare_with_baseline(self, baseline_file: str, output_dir: str = "results"):
        """Compare Reptile results with baseline results"""
        baseline_path = Path(baseline_file)

        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {baseline_file}")
            return

        # Load baseline results
        with open(baseline_path, "r") as f:
            baseline_results = json.load(f)

        # Run Reptile evaluation
        reptile_results = self.evaluate_comprehensive(output_dir)

        # Compare results
        comparison = self._create_comparison(baseline_results, reptile_results)

        # Save comparison
        output_path = Path(output_dir)
        model_name = self.config.gemma_model.split("/")[-1]
        comparison_file = output_path / f"baseline_vs_reptile_{model_name}.json"

        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"Comparison results saved to {comparison_file}")
        return comparison

    def _create_comparison(self, baseline_results: Dict, reptile_results: Dict) -> Dict:
        """Create comparison between baseline and Reptile results"""
        comparison = {
            "baseline_model": baseline_results.get("model_name", "unknown"),
            "reptile_model": reptile_results["config"]["model"],
            "task_comparisons": {},
            "summary": {},
        }

        # Compare task-by-task
        baseline_scores = baseline_results.get("results", {})
        reptile_zero_shot = reptile_results["evaluations"]["zero_shot"]
        reptile_few_shot = reptile_results["evaluations"].get("few_shot_5", {})

        for task_type in set(baseline_scores.keys()) | set(reptile_zero_shot.keys()):
            baseline_score = baseline_scores.get(task_type, {}).get("bleu", 0.0)
            reptile_zero = reptile_zero_shot.get(task_type, 0.0)
            reptile_few = reptile_few_shot.get(task_type, 0.0)

            comparison["task_comparisons"][task_type] = {
                "baseline_zero_shot": baseline_score,
                "reptile_zero_shot": reptile_zero,
                "reptile_few_shot_5": reptile_few,
                "zero_shot_improvement": reptile_zero - baseline_score,
                "few_shot_improvement": reptile_few - baseline_score,
            }

        # Summary statistics
        improvements = [
            comp["few_shot_improvement"]
            for comp in comparison["task_comparisons"].values()
        ]
        zero_improvements = [
            comp["zero_shot_improvement"]
            for comp in comparison["task_comparisons"].values()
        ]

        comparison["summary"] = {
            "avg_few_shot_improvement": (
                sum(improvements) / len(improvements) if improvements else 0.0
            ),
            "avg_zero_shot_improvement": (
                sum(zero_improvements) / len(zero_improvements)
                if zero_improvements
                else 0.0
            ),
            "tasks_improved_few_shot": sum(1 for imp in improvements if imp > 0),
            "total_tasks": len(improvements),
        }

        return comparison

    def print_evaluation_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("REPTILE META-LEARNING EVALUATION SUMMARY")
        print("=" * 60)

        config = results["config"]
        print(f"Model: {config['model']}")
        print(f"Base languages: {', '.join(config['base_langs'])}")
        print(f"Target languages: {', '.join(config['target_langs'])}")

        # Print summary statistics
        if "summary" in results:
            print(f"\nSummary Statistics:")
            for eval_type, stats in results["summary"].items():
                print(
                    f"  {eval_type}: {stats['mean']:.3f} ± {(stats['max']-stats['min'])/2:.3f} ({stats['count']} tasks)"
                )

        # Print detailed results
        print(f"\nDetailed Results:")
        for eval_type, eval_results in results["evaluations"].items():
            if eval_results:
                print(f"\n{eval_type.replace('_', ' ').title()}:")
                for task_type, score in sorted(eval_results.items()):
                    print(f"  {task_type}: {score:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Reptile meta-learning for translation"
    )
    parser.add_argument(
        "--model", choices=["270m", "1b"], default="1b", help="Model size to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        default=str(paths.results_dir),
        help="Output directory",
    )
    parser.add_argument("--baseline_file", help="Baseline results file for comparison")
    parser.add_argument(
        "--support_size",
        type=int,
        default=5,
        help="Number of support examples for few-shot evaluation",
    )
    parser.add_argument(
        "--language_groups",
        nargs="+",
        default=["az_tr_en", "be_uk_en"],
        help="Language groups to use for evaluation (e.g., az_tr_en be_uk_en)",
    )

    args = parser.parse_args()

    # Load .env file
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_project = os.getenv("WANDB_PROJECT")

    # Map model choices to actual model names
    model_mapping = {"270m": "google/gemma-3-270m-it", "1b": "google/gemma-3-1b-it"}

    # Create configuration
    config = ReptileConfig(
        support_size=args.support_size, gemma_model=model_mapping[args.model]
    )

    # Initialize evaluator
    evaluator = ReptileEvaluator(
        config,
        token=token,
        language_groups=args.language_groups,
        wandb_api_key=wandb_api_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    try:
        if args.baseline_file:
            # Compare with baseline
            comparison = evaluator.compare_with_baseline(
                args.baseline_file, args.output_dir
            )
            if comparison:
                print(f"\nBaseline vs Reptile Comparison:")
                print(
                    f"Average few-shot improvement: {comparison['summary']['avg_few_shot_improvement']:.3f}"
                )
                print(
                    f"Tasks improved: {comparison['summary']['tasks_improved_few_shot']}/{comparison['summary']['total_tasks']}"
                )
        else:
            # Run standalone evaluation
            results = evaluator.evaluate_comprehensive(args.output_dir)
            evaluator.print_evaluation_summary(results)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
