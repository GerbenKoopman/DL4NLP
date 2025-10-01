"""
Reptile Meta-Learning Implementation for Gemma Translation
Implements the Reptile algorithm for few-shot adaptation in translation tasks
"""

import torch
import logging
import numpy as np
import wandb
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from evaluation import TranslationEvaluator
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl.trainer.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning"""

    gemma_model: str = "google/gemma-3-1b-it"
    tokenizer_model: str = "google/gemma-3-1b-it"
    output_dir: str = "results/qlora_finetuned"
    max_length: int = 512
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_r: int = 16


@dataclass
class ReptileConfig:
    """Configuration for Reptile meta-learning"""

    inner_steps: int = 5
    meta_steps: int = 100
    support_size: int = 5
    query_size: int = 3
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    gemma_model: str = "google/gemma-3-1b-it"
    max_length: int = 128
    meta_lr: float = 0.1  # Meta-learning rate
    adapter_mode: str = "all"  # "all", "az_en", "be_en"

    base_langs: List[str] = field(default_factory=lambda: ["az", "be", "en"])
    target_langs: List[str] = field(default_factory=lambda: ["tr", "uk"])
    qlora_config: QLoRAConfig = field(default_factory=QLoRAConfig)


class ReptileMetaLearner:
    """Reptile meta-learning implementation for Gemma translation using QLoRA."""

    def __init__(
        self,
        config: ReptileConfig,
        token: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        self.config = config
        self.token = token
        self.model, self.tokenizer = self._load_base_model_and_tokenizer()
        self.evaluator = TranslationEvaluator()
        self.wandb_api_key = wandb_api_key
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        if self.wandb_api_key:
            wandb.login(key=self.wandb_api_key)
        logger.info(f"Initialized Reptile meta-learner with QLoRA on {config.device}")

    def save_adapter(self, output_dir: str):
        """Save the trained LoRA adapter."""
        self.model.save_pretrained(output_dir)
        logger.info(f"Adapter saved to {output_dir}")

    def _load_base_model_and_tokenizer(self):
        """Load the base Gemma model and tokenizer with QLoRA configuration."""
        if torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.qlora_config.gemma_model,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="eager",
            dtype=dtype,
            token=self.token,
        )

        peft_config = LoraConfig(
            r=self.config.qlora_config.lora_r,
            lora_alpha=self.config.qlora_config.lora_alpha,
            lora_dropout=self.config.qlora_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)
        self.peft_config = peft_config

        # Freeze the base model
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.qlora_config.tokenizer_model, token=self.token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _format_prompt(self, example):
        """Formats a dataset example into a prompt for the model."""
        instruction = (
            f"Translate the following text from {example['source_lang']} to {example['target_lang']}. "
            f"Provide only the translated text, without any additional commentary or prefixes.\n\n"
            f'Text: "{example["source_text"]}"\n'
            f"Translation: {example['target_text']}"
        )
        return {"text": instruction}

    def create_task_episodes(
        self, tasks: List[Dict], task_type: str
    ) -> List[Tuple[List[Dict], List[Dict]]]:
        """Create support/query episodes for a specific task type"""
        task_examples = [task for task in tasks if task.get("task_type") == task_type]

        if len(task_examples) < self.config.support_size + self.config.query_size:
            logger.warning(
                f"Insufficient examples for {task_type}: {len(task_examples)}"
            )
            return []

        episodes = []
        # Create multiple episodes by shuffling the data
        for _ in range(3):  # Create 3 episodes per task type
            task_examples_np = np.array(task_examples)
            np.random.shuffle(task_examples_np)
            shuffled_examples = task_examples_np.tolist()
            support_set = shuffled_examples[: self.config.support_size]
            query_set = shuffled_examples[
                self.config.support_size : self.config.support_size
                + self.config.query_size
            ]
            if len(query_set) < self.config.query_size:
                continue
            episodes.append((support_set, query_set))

        return episodes

    def _inner_loop_step(
        self, support_examples: List[Dict], query_examples: List[Dict]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Performs the inner loop adaptation, evaluation, and returns weight differences.
        """
        temp_output_dir = "tmp/inner_loop_adapter"
        support_dataset = Dataset.from_list(support_examples).map(self._format_prompt)

        training_args = TrainingArguments(
            output_dir=temp_output_dir,
            per_device_train_batch_size=self.config.qlora_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.qlora_config.gradient_accumulation_steps,
            learning_rate=self.config.qlora_config.learning_rate,
            num_train_epochs=self.config.inner_steps,
            logging_steps=1,
            fp16=torch.cuda.is_available()
            and getattr(self.model.config, "dtype", None) == torch.float16,
            bf16=torch.cuda.is_available()
            and getattr(self.model.config, "dtype", None) == torch.bfloat16,
            save_strategy="no",
            report_to="wandb" if self.wandb_api_key else [],
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=support_dataset,
            peft_config=self.peft_config,
        )

        initial_adapter_state = {
            k: v.clone() for k, v in self.model.named_parameters() if "lora" in k
        }

        if support_examples:
            trainer.train()

        # Evaluate on the query set
        total_bleu = 0.0
        total_chrf = 0.0
        if query_examples:
            for query in query_examples:
                prompt = (
                    self._format_prompt(query)["text"].split("Translation:")[0]
                    + "Translation:"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(
                    self.config.device
                )
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_length,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                full_output = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                translation = full_output[len(prompt) :].strip()

                scores = self.evaluator.evaluate_translation(
                    translation, query["target_text"], ["bleu", "chrf"]
                )
                total_bleu += scores["bleu"]
                total_chrf += scores["chrf"]

            avg_bleu = total_bleu / len(query_examples)
            avg_chrf = total_chrf / len(query_examples)
            combined_score = avg_bleu * 0.6 + avg_chrf * 0.4
            detailed_scores = {
                "bleu": avg_bleu,
                "chrf": avg_chrf,
                "combined_score": combined_score,
            }
        else:
            detailed_scores = {"bleu": 0.0, "chrf": 0.0, "combined_score": 0.0}

        adapted_adapter_state = {
            k: v.clone() for k, v in self.model.named_parameters() if "lora" in k
        }

        # Revert model to its initial state before this episode
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.data.copy_(initial_adapter_state[name])

        # Calculate the difference in weights
        weight_diff = {
            name: adapted_adapter_state[name] - initial_adapter_state[name]
            for name in initial_adapter_state
        }

        return weight_diff, detailed_scores

    def adapt_and_evaluate(
        self, support_examples: List[Dict], query_examples: List[Dict]
    ) -> Dict[str, float]:
        """
        Adapt the model on the support set using QLoRA and evaluate on the query set.
        This function performs the inner loop of Reptile.
        """
        # This method is now a wrapper around _inner_loop_step for evaluation purposes
        _, scores = self._inner_loop_step(support_examples, query_examples)
        return scores

    def reptile_step(
        self, tasks: List[Dict], task_types: List[str]
    ) -> Dict[str, float]:
        """Perform one Reptile meta-learning step"""
        task_performances = {}
        accumulated_weight_diffs = {}
        total_performance = 0.0
        total_bleu = 0.0
        total_chrf = 0.0
        valid_tasks = 0
        episodes = []

        # Sample tasks for this meta-step
        sampled_tasks = np.random.choice(
            task_types, min(len(task_types), 4), replace=False
        )

        for task_type in sampled_tasks:
            episodes = self.create_task_episodes(tasks, task_type)
            if not episodes:
                continue

            task_performance = 0.0
            task_bleu = 0.0
            task_chrf = 0.0
            valid_episodes = 0

            for support_set, query_set in episodes:
                # Inner loop adaptation and get weight differences
                weight_diff, episode_scores = self._inner_loop_step(
                    support_set, query_set
                )

                # Accumulate weight differences
                if not accumulated_weight_diffs:
                    accumulated_weight_diffs = weight_diff
                else:
                    for name in accumulated_weight_diffs:
                        accumulated_weight_diffs[name] += weight_diff[name]

                # Evaluate performance on the query set after adaptation
                task_performance += episode_scores["combined_score"]
                task_bleu += episode_scores["bleu"]
                task_chrf += episode_scores["chrf"]
                valid_episodes += 1

            if valid_episodes > 0:
                avg_task_performance = task_performance / valid_episodes
                avg_task_bleu = task_bleu / valid_episodes
                avg_task_chrf = task_chrf / valid_episodes
                task_performances[f"{task_type}_performance"] = avg_task_performance
                task_performances[f"{task_type}_bleu"] = avg_task_bleu
                task_performances[f"{task_type}_chrf"] = avg_task_chrf
                total_performance += avg_task_performance
                total_bleu += avg_task_bleu
                total_chrf += avg_task_chrf
                valid_tasks += 1

        # Apply the Reptile meta-update after processing the batch of tasks
        if accumulated_weight_diffs and valid_tasks > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if "lora" in name and name in accumulated_weight_diffs:
                        # Average the accumulated differences
                        avg_diff = accumulated_weight_diffs[name] / (
                            valid_tasks * len(episodes)
                        )
                        # Reptile update rule
                        param.data += self.config.meta_lr * avg_diff

        if valid_tasks > 0:
            task_performances["meta_average_performance"] = (
                total_performance / valid_tasks
            )
            task_performances["meta_average_bleu"] = total_bleu / valid_tasks
            task_performances["meta_average_chrf"] = total_chrf / valid_tasks
        else:
            task_performances["meta_average_performance"] = 0.0
            task_performances["meta_average_bleu"] = 0.0
            task_performances["meta_average_chrf"] = 0.0

        return task_performances

    def train_meta_learning(
        self, train_tasks: List[Dict], test_tasks: List[Dict]
    ) -> Dict[str, List[float]]:
        """Train using Reptile meta-learning algorithm"""
        if self.wandb_api_key:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=asdict(self.config),
            )
        logger.info(f"Starting Reptile meta-learning with {len(train_tasks)} tasks")

        # Get unique task types from base languages only
        all_task_types = list(set(task["task_type"] for task in train_tasks))

        if self.config.adapter_mode == "az_en":
            base_task_types = [t for t in all_task_types if "az" in t or "en" in t]
        elif self.config.adapter_mode == "be_en":
            base_task_types = [t for t in all_task_types if "be" in t or "en" in t]
        else:  # all
            base_task_types = [
                tt
                for tt in all_task_types
                if self.config.base_langs
                and any(lang in tt for lang in self.config.base_langs)
            ]

        logger.info(f"Training on task types: {base_task_types}")

        # Initialize training history to store all metrics
        training_history = {}
        train_keys = [
            "meta_average_performance",
            "meta_average_bleu",
            "meta_average_chrf",
        ]
        for task_type in base_task_types:
            train_keys.extend(
                [
                    f"{task_type}_performance",
                    f"{task_type}_bleu",
                    f"{task_type}_chrf",
                ]
            )
        for key in train_keys:
            training_history[key] = []

        test_keys = [
            "test_avg_performance",
            "test_avg_bleu",
            "test_avg_chrf",
        ]
        # Assuming test tasks can be from any language pair
        all_possible_test_tasks = ["tr_en", "en_tr", "uk_en", "en_uk"]
        for task_type in all_possible_test_tasks:
            test_keys.extend(
                [
                    f"test_{task_type}_performance",
                    f"test_{task_type}_bleu",
                    f"test_{task_type}_chrf",
                ]
            )
        for key in test_keys:
            training_history[key] = []

        global_step = 0
        for meta_step in range(self.config.meta_steps):
            # Perform one Reptile step
            step_performances = self.reptile_step(train_tasks, base_task_types)

            # Record and log training performance
            train_metrics_to_log = {}
            for key, performance in step_performances.items():
                if key in training_history:
                    training_history[key].append(performance)
                train_metrics_to_log[f"train_{key}"] = performance

            if self.wandb_api_key:
                wandb.log(train_metrics_to_log, step=global_step)

            # Evaluate on test set for learning curve
            if meta_step % 10 == 0:
                test_performance = self.evaluate_on_test_set(test_tasks)
                test_metrics_to_log = {}

                if test_performance:
                    avg_test_bleu = float(
                        np.mean([v["bleu"] for v in test_performance.values()])
                    )
                    avg_test_chrf = float(
                        np.mean([v["chrf"] for v in test_performance.values()])
                    )
                    avg_test_perf = float(
                        np.mean(
                            [v["combined_score"] for v in test_performance.values()]
                        )
                    )

                    # Record average test scores
                    training_history["test_avg_performance"].append(avg_test_perf)
                    training_history["test_avg_bleu"].append(avg_test_bleu)
                    training_history["test_avg_chrf"].append(avg_test_chrf)

                    # Log average test scores
                    test_metrics_to_log["test_avg_performance"] = avg_test_perf
                    test_metrics_to_log["test_avg_bleu"] = avg_test_bleu
                    test_metrics_to_log["test_avg_chrf"] = avg_test_chrf

                    # Record and log individual test task scores
                    for task, scores in test_performance.items():
                        for metric, value in scores.items():
                            history_key = (
                                f"test_{task}_{metric.replace('combined_', '')}"
                            )
                            if history_key in training_history:
                                training_history[history_key].append(value)
                            log_key = f"test_{task}_{metric.replace('combined_', '')}"
                            test_metrics_to_log[log_key] = value

                    logger.info(f"Test Performance: {avg_test_perf:.3f}")

                else:
                    logger.info("Test Performance: No test tasks evaluated.")

                if self.wandb_api_key and test_metrics_to_log:
                    wandb.log(test_metrics_to_log, step=global_step)

            # Log progress
            if meta_step % 10 == 0:
                avg_perf = step_performances.get("meta_average_performance", 0.0)
                logger.info(
                    f"Meta-step {meta_step}/{self.config.meta_steps}, "
                    f"Avg Performance: {avg_perf:.3f}"
                )

            global_step += 1

        if self.wandb_api_key:
            wandb.finish()
        logger.info("Completed Reptile meta-learning training")
        return training_history

    def evaluate_on_test_set(
        self, test_tasks: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance on a held-out test set based on adapter mode."""
        logger.info("Evaluating on test set...")
        test_performance = {}

        target_test_tasks = []
        if self.config.adapter_mode == "az_en":
            target_test_tasks = ["tr_en", "en_tr"]
        elif self.config.adapter_mode == "be_en":
            target_test_tasks = ["uk_en", "en_uk"]
        elif self.config.adapter_mode == "all":
            target_test_tasks = ["tr_en", "en_tr", "uk_en", "en_uk"]

        filtered_test_tasks = [
            task for task in test_tasks if task["task_type"] in target_test_tasks
        ]
        test_task_types = list(set(task["task_type"] for task in filtered_test_tasks))
        logger.info(f"Evaluating on test task types: {test_task_types}")

        for task_type in test_task_types:
            episodes = self.create_task_episodes(filtered_test_tasks, task_type)
            if not episodes:
                continue

            task_scores_list = []
            for support_set, query_set in episodes:
                scores = self._inner_loop_step(support_set, query_set)[1]
                task_scores_list.append(scores)

            if task_scores_list:
                avg_bleu = np.mean([s["bleu"] for s in task_scores_list])
                avg_chrf = np.mean([s["chrf"] for s in task_scores_list])
                avg_combined = np.mean([s["combined_score"] for s in task_scores_list])
                test_performance[task_type] = {
                    "bleu": avg_bleu,
                    "chrf": avg_chrf,
                    "combined_score": avg_combined,
                }
                logger.debug(
                    f"Test score for {task_type}: {avg_combined:.3f} (BLEU: {avg_bleu:.3f}, CHRF: {avg_chrf:.3f})"
                )

        return test_performance

    def evaluate_transfer(
        self, test_tasks: List[Dict], num_shots: int = 5
    ) -> Dict[str, float]:
        """Evaluate transfer to target languages (Turkish, Ukrainian)"""
        results = {}

        # Filter test tasks for target languages
        target_task_types = [
            task["task_type"]
            for task in test_tasks
            if self.config.target_langs
            and any(lang in task["task_type"] for lang in self.config.target_langs)
        ]
        unique_target_tasks = list(set(target_task_types))

        logger.info(f"Evaluating transfer on: {unique_target_tasks}")

        for task_type in unique_target_tasks:
            task_examples = [
                task for task in test_tasks if task["task_type"] == task_type
            ]

            if len(task_examples) < num_shots + 1:
                logger.warning(
                    f"Insufficient examples for {task_type}: {len(task_examples)}"
                )
                continue

            # Split into support and query
            support_examples = task_examples[:num_shots]
            query_examples = task_examples[
                num_shots : num_shots + 5
            ]  # Test on 5 examples

            # Adapt and evaluate
            performance = self.adapt_and_evaluate(support_examples, query_examples)
            results[task_type] = performance["combined_score"]

            logger.info(
                f"Transfer performance for {task_type}: {performance['combined_score']:.3f}"
            )

        return results

    def zero_shot_evaluate(self, test_tasks: List[Dict]) -> Dict[str, float]:
        """Evaluate zero-shot performance without adaptation"""
        results = {}
        task_types = list(set(task["task_type"] for task in test_tasks))

        for task_type in task_types:
            task_examples = [
                task for task in test_tasks if task["task_type"] == task_type
            ]
            if not task_examples:
                continue

            # Evaluate on a few examples without a support set
            query_examples = task_examples[:5]
            performance = self.adapt_and_evaluate([], query_examples)
            results[task_type] = performance["combined_score"]
            logger.info(
                f"Zero-shot performance for {task_type}: {performance['combined_score']:.3f}"
            )

        return results


def main():
    """Test Reptile meta-learning"""
    # Example usage
    config = ReptileConfig(meta_steps=20, inner_steps=3)
    meta_learner = ReptileMetaLearner(config)

    # Create dummy tasks for testing
    dummy_tasks = []
    for i in range(20):
        dummy_tasks.append(
            {
                "source_text": f"Hello world {i}",
                "target_text": f"Salam dünya {i}",
                "source_lang": "en",
                "target_lang": "az",
                "task_type": "en_az",
            }
        )
        dummy_tasks.append(
            {
                "source_text": f"Good morning {i}",
                "target_text": f"Dobry ranak {i}",
                "source_lang": "en",
                "target_lang": "be",
                "task_type": "en_be",
            }
        )

    # Test meta-learning
    logger.info("Testing Reptile meta-learning...")
    history = meta_learner.train_meta_learning(dummy_tasks, dummy_tasks)
    print("Training history:", history)

    # Test zero-shot evaluation
    zero_shot_results = meta_learner.zero_shot_evaluate(dummy_tasks)
    print("Zero-shot results:", zero_shot_results)


if __name__ == "__main__":
    main()
