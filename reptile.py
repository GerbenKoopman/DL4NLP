"""
Reptile Meta-Learning Implementation for Gemma Translation
Implements the Reptile algorithm for few-shot adaptation in translation tasks
"""

import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma import GemmaTranslationModel
from evaluation import TranslationEvaluator
from cache import get_cached_gemma_model

logger = logging.getLogger(__name__)


@dataclass
class ReptileConfig:
    """Configuration for Reptile meta-learning"""

    inner_steps: int = 5  # Number of inner loop adaptation steps
    meta_steps: int = 100  # Number of meta-learning episodes
    support_size: int = 5  # Number of support examples per task
    query_size: int = 3  # Number of query examples per task
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    gemma_model: str = "google/gemma-3-270m-it"  # Gemma model name
    max_length: int = 128  # Max generation length
    temperature: float = 0.7  # Generation temperature

    # Language configuration for project
    base_langs: List[str] | None = None  # Azerbaijani, Belarusian, English
    target_langs: List[str] | None = None  # Turkish, Ukrainian

    def __post_init__(self):
        if self.base_langs is None:
            self.base_langs = ["az", "be", "en"]
        if self.target_langs is None:
            self.target_langs = ["tr", "uk"]


class ReptileMetaLearner:
    """Reptile meta-learning implementation for Gemma translation"""

    def __init__(self, config: ReptileConfig, token: Optional[str] = None):
        self.config = config
        self.gemma_model = get_cached_gemma_model(
            config.gemma_model, config.device, token=token
        )
        self.evaluator = TranslationEvaluator()

        # Track adaptation context (in-context learning examples)
        self.adaptation_context = {}

        logger.info(f"Initialized Reptile meta-learner with Gemma on {config.device}")

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
            np.random.shuffle(task_examples)
            support_set = task_examples[: self.config.support_size]
            query_set = task_examples[
                self.config.support_size : self.config.support_size
                + self.config.query_size
            ]
            episodes.append((support_set, query_set))

        return episodes

    def adapt_in_context(self, support_examples: List[Dict]) -> List[Dict]:
        """
        Reptile adaptation through in-context learning
        For Gemma, this means selecting the best few-shot examples
        """
        # Convert to format expected by Gemma
        few_shot_examples = []
        for ex in support_examples:
            few_shot_examples.append(
                {
                    "source": ex["source_text"],
                    "target": ex["target_text"],
                    "source_lang": ex["source_lang"],
                    "target_lang": ex["target_lang"],
                }
            )

        return few_shot_examples

    def evaluate_query_set(
        self, query_examples: List[Dict], adaptation_context: List[Dict]
    ) -> float:
        """Evaluate performance on query set using adapted context"""
        total_score = 0.0
        valid_queries = 0

        for query in query_examples:
            try:
                # Generate translation using adapted context
                translation = self.gemma_model.translate(
                    query["source_text"],
                    query["source_lang"],
                    query["target_lang"],
                    few_shot_examples=adaptation_context,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                )

                # Evaluate translation quality
                scores = self.evaluator.evaluate_translation(
                    translation, query["target_text"], ["bleu", "chrf"]
                )

                # Combine scores (weighted average)
                combined_score = scores["bleu"] * 0.6 + scores["chrf"] * 0.4
                total_score += combined_score
                valid_queries += 1

            except Exception as e:
                logger.warning(f"Query evaluation failed: {e}")
                continue

        return total_score / valid_queries if valid_queries > 0 else 0.0

    def reptile_step(
        self, tasks: List[Dict], task_types: List[str]
    ) -> Dict[str, float]:
        """Perform one Reptile meta-learning step"""
        task_performances = {}
        total_performance = 0.0
        valid_tasks = 0

        # Sample tasks for this meta-step
        sampled_tasks = np.random.choice(
            task_types, min(len(task_types), 4), replace=False
        )

        for task_type in sampled_tasks:
            episodes = self.create_task_episodes(tasks, task_type)

            if not episodes:
                continue

            task_performance = 0.0
            valid_episodes = 0

            for support_set, query_set in episodes:
                # Inner loop: adapt using support set
                adaptation_context = self.adapt_in_context(support_set)

                # Evaluate on query set
                episode_score = self.evaluate_query_set(query_set, adaptation_context)

                task_performance += episode_score
                valid_episodes += 1

            if valid_episodes > 0:
                avg_task_performance = task_performance / valid_episodes
                task_performances[task_type] = avg_task_performance
                total_performance += avg_task_performance
                valid_tasks += 1

        # For Gemma, we don't update parameters but track performance
        avg_performance = total_performance / valid_tasks if valid_tasks > 0 else 0.0
        task_performances["meta_average"] = avg_performance

        return task_performances

    def train_meta_learning(self, tasks: List[Dict]) -> Dict[str, List[float]]:
        """Train using Reptile meta-learning algorithm"""
        logger.info(f"Starting Reptile meta-learning with {len(tasks)} tasks")

        # Get unique task types from base languages only
        all_task_types = list(set(task["task_type"] for task in tasks))
        base_task_types = [
            tt
            for tt in all_task_types
            if any(lang in tt for lang in self.config.base_langs)
        ]

        logger.info(f"Training on task types: {base_task_types}")

        training_history = {task_type: [] for task_type in base_task_types}
        training_history["meta_average"] = []

        for meta_step in range(self.config.meta_steps):
            # Perform one Reptile step
            step_performances = self.reptile_step(tasks, base_task_types)

            # Record performance
            for task_type, performance in step_performances.items():
                if task_type in training_history:
                    training_history[task_type].append(performance)

            # Log progress
            if meta_step % 10 == 0:
                avg_perf = step_performances.get("meta_average", 0.0)
                logger.info(
                    f"Meta-step {meta_step}/{self.config.meta_steps}, "
                    f"Avg Performance: {avg_perf:.3f}"
                )

        logger.info("Completed Reptile meta-learning training")
        return training_history

    def evaluate_transfer(
        self, test_tasks: List[Dict], num_shots: int = 5
    ) -> Dict[str, float]:
        """Evaluate transfer to target languages (Turkish, Ukrainian)"""
        results = {}

        # Filter test tasks for target languages
        target_task_types = [
            task["task_type"]
            for task in test_tasks
            if any(lang in task["task_type"] for lang in self.config.target_langs)
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

            # Adapt using support examples
            adaptation_context = self.adapt_in_context(support_examples)

            # Evaluate on query examples
            performance = self.evaluate_query_set(query_examples, adaptation_context)
            results[task_type] = performance

            logger.info(f"Transfer performance for {task_type}: {performance:.3f}")

        return results

    def zero_shot_evaluate(self, test_tasks: List[Dict]) -> Dict[str, float]:
        """Evaluate zero-shot performance without adaptation"""
        results = {}

        task_types = list(set(task["task_type"] for task in test_tasks))

        for task_type in task_types:
            task_examples = [
                task for task in test_tasks if task["task_type"] == task_type
            ]

            if len(task_examples) < 3:
                continue

            # Use first 3 examples for zero-shot evaluation
            query_examples = task_examples[:3]

            # Evaluate without any adaptation context
            performance = self.evaluate_query_set(query_examples, [])
            results[task_type] = performance

            logger.info(f"Zero-shot performance for {task_type}: {performance:.3f}")

        return results


def main():
    """Test Reptile meta-learning"""
    # Example usage
    config = ReptileConfig(meta_steps=20, inner_steps=3)
    meta_learner = ReptileMetaLearner(config)

    # Create dummy tasks for testing
    dummy_tasks = [
        {
            "source_text": "Hello world",
            "target_text": "Salam dünya",
            "source_lang": "en",
            "target_lang": "az",
            "task_type": "en_az",
        },
        {
            "source_text": "Good morning",
            "target_text": "Dobry ranak",
            "source_lang": "en",
            "target_lang": "be",
            "task_type": "en_be",
        },
    ] * 20  # Repeat to have enough examples

    # Test meta-learning
    logger.info("Testing Reptile meta-learning...")
    history = meta_learner.train_meta_learning(dummy_tasks)

    # Test zero-shot evaluation
    zero_shot_results = meta_learner.zero_shot_evaluate(dummy_tasks)
    print("Zero-shot results:", zero_shot_results)


if __name__ == "__main__":
    main()
