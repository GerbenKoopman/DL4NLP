"""
Ablation Study Experiment Runner
Systematically runs ablation experiments across multiple configurations
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product
import pandas as pd

from paths import paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AblationStudyRunner:
    """Run systematic ablation experiments"""

    def __init__(self, model: str = "1b", output_dir: str = None):
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else paths.results_dir / "ablation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_ablation_configs(self, ablation_type: str = "minimal") -> List[Dict]:
        """Generate ablation configurations based on type
        
        Args:
            ablation_type: "minimal" (16 configs), "extended" (more comprehensive), "full" (all combinations)
        """
        if ablation_type == "minimal":
            # Core ablation: 2x2x2x2 = 16 configs
            configs = []
            for meta_lr, inner_steps, adapter_mode, support_size in product(
                [0.0, 0.1],  # meta_lr: no update vs standard
                [0, 3],  # inner_steps: no adaptation vs moderate
                ["az_en", "be_en"],  # adapter_mode: single family
                [1, 5],  # support_size: 1-shot vs 5-shot
            ):
                configs.append({
                    "meta_lr": meta_lr,
                    "inner_steps": inner_steps,
                    "adapter_mode": adapter_mode,
                    "support_size": support_size,
                    "query_size": 3,
                    "meta_steps": 50,  # Reduced for ablation
                    "bleu_weight": 0.6,
                    "seed": 42,
                    "episodes_per_task": 3,
                })
            return configs
            
        elif ablation_type == "extended":
            # Extended ablation: include "all" mode and metric variations
            configs = []
            for meta_lr, inner_steps, adapter_mode, support_size, bleu_weight in product(
                [0.0, 0.05, 0.1],
                [0, 1, 3, 5],
                ["az_en", "be_en", "all"],
                [1, 5],
                [0.5, 0.6],
            ):
                configs.append({
                    "meta_lr": meta_lr,
                    "inner_steps": inner_steps,
                    "adapter_mode": adapter_mode,
                    "support_size": support_size,
                    "query_size": 3,
                    "meta_steps": 50,
                    "bleu_weight": bleu_weight,
                    "seed": 42,
                    "episodes_per_task": 3,
                })
            return configs
            
        elif ablation_type == "metric_only":
            # Test metric weight sensitivity only
            configs = []
            for bleu_weight in [0.4, 0.5, 0.6, 0.7]:
                configs.append({
                    "meta_lr": 0.1,
                    "inner_steps": 3,
                    "adapter_mode": "all",
                    "support_size": 5,
                    "query_size": 3,
                    "meta_steps": 50,
                    "bleu_weight": bleu_weight,
                    "seed": 42,
                    "episodes_per_task": 3,
                })
            return configs
            
        elif ablation_type == "scaling":
            # For testing different model sizes with best config
            configs = []
            configs.append({
                "meta_lr": 0.1,
                "inner_steps": 3,
                "adapter_mode": "all",
                "support_size": 5,
                "query_size": 3,
                "meta_steps": 50,
                "bleu_weight": 0.6,
                "seed": 42,
                "episodes_per_task": 3,
            })
            return configs
            
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    def generate_config_name(self, config: Dict) -> str:
        """Generate descriptive name for configuration"""
        return (
            f"metaLR{config['meta_lr']:.2f}_"
            f"inner{config['inner_steps']}_"
            f"support{config['support_size']}_"
            f"{config['adapter_mode']}_"
            f"bleu{config['bleu_weight']:.2f}_"
            f"seed{config['seed']}"
        )
    
    def run_single_experiment(
        self, config: Dict, config_name: str, language_groups: List[str]
    ) -> Tuple[bool, str]:
        """Run a single training experiment with given config"""
        logger.info(f"Running experiment: {config_name}")
        
        # Build command
        cmd = [
            sys.executable,
            str(paths.base_dir / "src" / "train_reptile.py"),
            "--model", self.model,
            "--meta_steps", str(config["meta_steps"]),
            "--inner_steps", str(config["inner_steps"]),
            "--support_size", str(config["support_size"]),
            "--query_size", str(config["query_size"]),
            "--adapter_mode", config["adapter_mode"],
            "--meta_lr", str(config["meta_lr"]),
            "--bleu_weight", str(config["bleu_weight"]),
            "--seed", str(config["seed"]),
            "--episodes_per_task", str(config["episodes_per_task"]),
            "--output_dir", str(self.output_dir / config_name),
            "--language_groups", *language_groups,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=7200,  # 2 hour timeout per experiment
            )
            logger.info(f"✓ Completed: {config_name}")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed: {config_name}")
            logger.error(f"Error: {e.stderr}")
            return False, e.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout: {config_name}")
            return False, "Timeout after 2 hours"
    
    def run_evaluation(
        self, config_name: str, language_groups: List[str]
    ) -> Tuple[bool, str]:
        """Run evaluation for a trained model"""
        logger.info(f"Evaluating: {config_name}")
        
        cmd = [
            sys.executable,
            str(paths.base_dir / "src" / "evaluate_reptile.py"),
            "--model", self.model,
            "--adapter_mode", config["adapter_mode"],
            "--output_dir", str(self.output_dir / config_name),
            "--language_groups", *language_groups,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout
            )
            logger.info(f"✓ Evaluation completed: {config_name}")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Evaluation failed: {config_name}")
            logger.error(f"Error: {e.stderr}")
            return False, e.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Evaluation timeout: {config_name}")
            return False, "Timeout after 1 hour"
    
    def run_ablation_study(
        self,
        ablation_type: str = "minimal",
        language_groups: List[str] = None,
        skip_training: bool = False,
        skip_evaluation: bool = False,
    ):
        """Run complete ablation study"""
        if language_groups is None:
            language_groups = ["az_tr_en", "be_uk_en"]
        
        configs = self.generate_ablation_configs(ablation_type)
        logger.info(f"Running {ablation_type} ablation study with {len(configs)} configurations")
        
        results = {
            "ablation_type": ablation_type,
            "model": self.model,
            "language_groups": language_groups,
            "experiments": [],
        }
        
        for i, config in enumerate(configs, 1):
            config_name = self.generate_config_name(config)
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i}/{len(configs)}: {config_name}")
            logger.info(f"{'='*60}")
            
            experiment_result = {
                "config_name": config_name,
                "config": config,
                "training_success": False,
                "evaluation_success": False,
            }
            
            # Training
            if not skip_training:
                train_success, train_output = self.run_single_experiment(
                    config, config_name, language_groups
                )
                experiment_result["training_success"] = train_success
                
                if not train_success:
                    logger.warning(f"Skipping evaluation for {config_name} due to training failure")
                    results["experiments"].append(experiment_result)
                    continue
            else:
                experiment_result["training_success"] = True  # Assume exists
            
            # Evaluation
            if not skip_evaluation:
                eval_success, eval_output = self.run_evaluation(
                    config_name, language_groups
                )
                experiment_result["evaluation_success"] = eval_success
            
            results["experiments"].append(experiment_result)
            
            # Save progress after each experiment
            self.save_ablation_summary(results)
        
        logger.info(f"\n{'='*60}")
        logger.info("Ablation study completed!")
        logger.info(f"{'='*60}")
        self.print_ablation_summary(results)
        
        return results
    
    def save_ablation_summary(self, results: Dict):
        """Save ablation study summary"""
        summary_file = self.output_dir / f"ablation_summary_{self.model}.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")
    
    def print_ablation_summary(self, results: Dict):
        """Print summary of ablation study"""
        total = len(results["experiments"])
        successful_training = sum(
            1 for exp in results["experiments"] if exp["training_success"]
        )
        successful_eval = sum(
            1 for exp in results["experiments"] if exp["evaluation_success"]
        )
        
        print(f"\nAblation Study Summary:")
        print(f"  Type: {results['ablation_type']}")
        print(f"  Model: {results['model']}")
        print(f"  Total experiments: {total}")
        print(f"  Successful training: {successful_training}/{total}")
        print(f"  Successful evaluation: {successful_eval}/{total}")
        
        if successful_training < total:
            print(f"\n  Failed experiments:")
            for exp in results["experiments"]:
                if not exp["training_success"]:
                    print(f"    - {exp['config_name']}")
    
    def aggregate_results(self) -> pd.DataFrame:
        """Aggregate results from all experiments into a DataFrame"""
        all_results = []
        
        for exp_dir in self.output_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Find evaluation results
            eval_file = exp_dir / f"reptile_evaluation_gemma-3-{self.model}-it.json"
            if not eval_file.exists():
                logger.warning(f"No evaluation results for {exp_dir.name}")
                continue
            
            with open(eval_file) as f:
                eval_data = json.load(f)
            
            # Extract config from directory name
            config_name = exp_dir.name
            
            # Parse evaluation results
            for eval_type, task_results in eval_data.get("evaluations", {}).items():
                for task_type, score in task_results.items():
                    all_results.append({
                        "config_name": config_name,
                        "evaluation_type": eval_type,
                        "task_type": task_type,
                        "score": score,
                    })
        
        if not all_results:
            logger.warning("No results found to aggregate")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        output_file = self.output_dir / f"aggregated_results_{self.model}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Aggregated results saved to {output_file}")
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study experiments"
    )
    parser.add_argument(
        "--model",
        choices=["270m", "1b", "4b"],
        default="1b",
        help="Model size for ablation study",
    )
    parser.add_argument(
        "--ablation_type",
        choices=["minimal", "extended", "metric_only", "scaling"],
        default="minimal",
        help="Type of ablation study to run",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: results/ablation)",
    )
    parser.add_argument(
        "--language_groups",
        nargs="+",
        default=["az_tr_en", "be_uk_en"],
        help="Language groups to use",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training (only run evaluation on existing models)",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation (only run training)",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate existing results (no training/eval)",
    )
    
    args = parser.parse_args()
    
    runner = AblationStudyRunner(
        model=args.model,
        output_dir=args.output_dir,
    )
    
    if args.aggregate_only:
        logger.info("Aggregating results only...")
        df = runner.aggregate_results()
        if not df.empty:
            print(f"\nAggregated {len(df)} result entries")
    else:
        runner.run_ablation_study(
            ablation_type=args.ablation_type,
            language_groups=args.language_groups,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
        )
        
        # Aggregate results after completion
        logger.info("\nAggregating all results...")
        runner.aggregate_results()


if __name__ == "__main__":
    main()

