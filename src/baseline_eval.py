"""
Baseline Evaluation Script
Zero-shot evaluation of Gemma models on translation tasks
"""

import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from datasets import Dataset
import torch
import evaluate
import numpy as np

from gemma import GemmaTranslationModel
from paths import paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluate Gemma models on zero-shot translation tasks"""

    def __init__(
        self,
        model_name: str,
        language_groups: List[str],
        token: Optional[str] = None,
        device: Optional[str] = None,
        max_eval_samples: Optional[int] = None,
        eval_pair_policy: str = "target",
    ):
        self.model_name = model_name
        self.language_groups = language_groups
        self.gemma = GemmaTranslationModel(
            model_name=self.model_name,
            token=token,
            use_lora=False,  # No LoRA for baseline
            device=device,
        )
        self.gemma.load_model()

        self.max_eval_samples = max_eval_samples
        self.eval_pair_policy = eval_pair_policy

        # Initialize metrics
        self.chrf = evaluate.load("chrf")
        self.bleu = evaluate.load("sacrebleu")

    def _load_and_prepare_data(self, split: str, pair_policy: str = "all") -> Dataset:
        """Load data and format it into a Hugging Face Dataset."""
        all_examples = []
        for group in self.language_groups:
            data_file = paths.data_dir / f"{group}_{split}.pkl"
            if not data_file.exists():
                logger.warning(f"Data file not found: {data_file}. Skipping.")
                continue

            logger.info(f"Loading {split} data from {data_file}")
            with open(data_file, "rb") as f:
                df = pickle.load(f)
                for _, row in df.iterrows():
                    langs = [col for col in df.columns if col not in ["talk_name"]]
                    for src_lang in langs:
                        for tgt_lang in langs:
                            if src_lang == tgt_lang:
                                continue
                            keep = False
                            if pair_policy == "all":
                                keep = True
                            elif pair_policy == "base":
                                keep = src_lang == "en" or tgt_lang == "en"
                            elif pair_policy == "target":
                                target_set = None
                                if "az_tr" in group:
                                    target_set = {"az", "tr"}
                                elif "be_uk" in group:
                                    target_set = {"be", "uk"}
                                if target_set is not None:
                                    keep = (
                                        src_lang in target_set
                                        and tgt_lang in target_set
                                    )
                            if keep:
                                all_examples.append(
                                    {
                                        "source_text": row[src_lang],
                                        "target_text": row[tgt_lang],
                                        "source_lang": src_lang,
                                        "target_lang": tgt_lang,
                                    }
                                )
        if not all_examples:
            raise FileNotFoundError(f"No {split} data could be loaded.")

        return Dataset.from_list(all_examples)

    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, float]:
        """Evaluate zero-shot performance on a given dataset."""
        if self.gemma.model is None:
            logger.error("Model not loaded. Aborting evaluation.")
            return {"bleu": 0.0, "chrf": 0.0}

        logger.info(f"Evaluating {len(dataset)} examples (zero-shot)...")

        translations = []
        references = []

        # Access columns as lists for iteration
        source_texts = dataset["source_text"]
        source_langs = dataset["source_lang"]
        target_langs = dataset["target_lang"]
        target_texts = dataset["target_text"]

        for i in range(len(dataset)):
            try:
                translation = self.gemma.translate(
                    source_texts[i],
                    source_langs[i],
                    target_langs[i],
                    max_length=256,
                )
                translations.append(translation)
                references.append(target_texts[i])

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} examples")

            except Exception as e:
                logger.error(f"Translation failed for example {i}: {e}", exc_info=True)
                translations.append("")
                references.append(target_texts[i])

        # Compute metrics
        chrf_score = self.chrf.compute(predictions=translations, references=references)
        bleu_score = self.bleu.compute(
            predictions=translations, references=[[r] for r in references]
        )

        result = {
            "chrf": chrf_score["score"] if chrf_score else 0.0,
            "bleu": bleu_score["score"] if bleu_score else 0.0,
        }
        logger.info(f"Evaluation results: {result}")
        return result

    def run_baseline_evaluation(self, output_dir: str):
        """Run complete baseline evaluation"""
        logger.info("Starting baseline evaluation.")

        # Evaluate on the dev set
        logger.info("Evaluating on the dev set.")
        dev_dataset = self._load_and_prepare_data(
            "dev", pair_policy=self.eval_pair_policy
        )
        if (
            self.max_eval_samples is not None
            and len(dev_dataset) > self.max_eval_samples
        ):
            dev_dataset = dev_dataset.select(range(self.max_eval_samples))
            logger.info(f"Subsampled dev dataset to {len(dev_dataset)} examples")

        dev_results = self.evaluate_dataset(dev_dataset)
        logger.info(f"Final evaluation results on dev set: {dev_results}")

        dev_results_file = Path(output_dir) / "baseline_dev_evaluation.json"
        with open(dev_results_file, "w", encoding="utf-8") as f:
            json.dump(dev_results, f, indent=2)
        logger.info(f"Dev results saved to {dev_results_file}")

        # Evaluate on the test set
        logger.info("Evaluating on the test set.")
        test_dataset = self._load_and_prepare_data(
            "test", pair_policy=self.eval_pair_policy
        )
        if (
            self.max_eval_samples is not None
            and len(test_dataset) > self.max_eval_samples
        ):
            test_dataset = test_dataset.select(range(self.max_eval_samples))
            logger.info(f"Subsampled test dataset to {len(test_dataset)} examples")

        test_results = self.evaluate_dataset(test_dataset)
        logger.info(f"Test evaluation results: {test_results}")

        test_results_file = Path(output_dir) / "baseline_test_evaluation.json"
        with open(test_results_file, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Test results saved to {test_results_file}")

        self.print_summary(dev_results, test_results)

    def print_summary(self, dev_results: Dict, test_results: Dict):
        """Print evaluation summary."""
        model_short_name = self.model_name.split("/")[-1]
        print("\n" + "=" * 60)
        print(f"BASELINE EVALUATION SUMMARY - {model_short_name}")
        print("=" * 60)
        print(
            f"Policy: {self.eval_pair_policy}, Max Samples: {self.max_eval_samples or 'All'}"
        )

        print("\nDev Set Results:")
        print(f"  BLEU: {dev_results.get('bleu', 0.0):.2f}")
        print(f"  chrF: {dev_results.get('chrf', 0.0):.2f}")

        print("\nTest Set Results:")
        print(f"  BLEU: {test_results.get('bleu', 0.0):.2f}")
        print(f"  chrF: {test_results.get('chrf', 0.0):.2f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation of Gemma models, mirroring train_lora.py setup."
    )
    parser.add_argument(
        "--model", choices=["270m", "1b"], default="1b", help="Model size to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        default=str(paths.results_dir),
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--language_groups",
        nargs="+",
        default=["az_tr_en", "be_uk_en"],
        help="Language groups to use for evaluation (e.g., az_tr_en be_uk_en).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Force a device (cpu/mps/cuda). If omitted, gemma decides.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit number of eval/test examples (for quick local sanity checks).",
    )
    parser.add_argument(
        "--eval_pair_policy",
        choices=["all", "base", "target"],
        default="target",
        help="Which direction pairs to include for EVAL/TEST (default: target=within related languages).",
    )
    args = parser.parse_args()

    # Load .env file
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    model_mapping = {"270m": "google/gemma-3-270m-it", "1b": "google/gemma-3-1b-it"}
    model_name = model_mapping[args.model]

    # Create output directory
    output_dir = Path(args.output_dir) / f"baseline_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run evaluator
    evaluator = BaselineEvaluator(
        model_name=model_name,
        language_groups=args.language_groups,
        token=token,
        device=args.device,
        max_eval_samples=args.max_eval_samples,
        eval_pair_policy=args.eval_pair_policy,
    )

    try:
        evaluator.run_baseline_evaluation(str(output_dir))
        logger.info("Baseline evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
