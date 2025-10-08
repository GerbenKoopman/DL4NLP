"""
LoRA Fine-tuning Script
Fine-tune Gemma models using LoRA for translation tasks.
"""

import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import List, Optional
import os
from dotenv import load_dotenv
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

from gemma import GemmaTranslationModel
from paths import paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoRAFinetuner:
    """Fine-tune Gemma models using LoRA."""

    def __init__(
        self,
        model_name: str,
        language_groups: List[str],
        token: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ):
        self.model_name = model_name
        self.language_groups = language_groups
        self.gemma = GemmaTranslationModel(
            model_name=self.model_name,
            token=token,
            use_lora=True,
        )
        self.gemma.load_model()

        # Initialize metrics
        self.chrf = evaluate.load("chrf")
        self.bleu = evaluate.load("sacrebleu")

        # Set up wandb if API key is provided
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            os.environ["WANDB_PROJECT"] = wandb_project or "lora-finetuning"
            os.environ["WANDB_LOG_MODEL"] = "epoch"

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def _load_and_prepare_data(self, split: str) -> Dataset:
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
                            if src_lang != tgt_lang:
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

    def _preprocess_function(self, examples):
        """Tokenize the input and target texts."""
        prompts = [
            self.gemma._create_prompt(text, src, tgt)
            for text, src, tgt in zip(
                examples["source_text"],
                examples["source_lang"],
                examples["target_lang"],
            )
        ]

        # The entire sequence (prompt + target) is used for training
        full_texts = [
            prompt + target for prompt, target in zip(prompts, examples["target_text"])
        ]

        if self.gemma.tokenizer is None:
            raise ValueError(
                "Tokenizer is None. Check GemmaTranslationModel initialization."
            )

        model_inputs = self.gemma.tokenizer(
            full_texts,
            max_length=256,
            padding="max_length",
            truncation=True,
        )

        # The labels are the same as the input_ids for language modeling
        model_inputs["labels"] = model_inputs["input_ids"]

        return model_inputs

    def compute_metrics(self, eval_pred):
        """Computes CHRF and BLEU scores for evaluation."""
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if self.gemma.tokenizer is None:
            raise ValueError("Tokenizer is not available for computing metrics.")

        # Decode generated summaries, which are in preds
        decoded_preds = self.gemma.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Decode labels
        labels = np.where(labels != -100, labels, self.gemma.tokenizer.pad_token_id)
        decoded_labels = self.gemma.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # Compute metrics
        chrf_score = self.chrf.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        bleu_score = self.bleu.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        result = {
            "chrf": chrf_score["score"] if chrf_score else 0.0,
            "bleu": bleu_score["score"] if bleu_score else 0.0,
        }

        return result

    def train(
        self,
        output_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        eval_batch_size: int,
        gradient_accumulation_steps: int,
    ):
        """Run the LoRA fine-tuning process."""
        logger.info("Starting LoRA fine-tuning.")

        # Load and preprocess data
        train_dataset = self._load_and_prepare_data("train")
        tokenized_train_dataset = train_dataset.map(
            self._preprocess_function, batched=True
        )

        eval_dataset = self._load_and_prepare_data("dev")
        tokenized_eval_dataset = eval_dataset.map(
            self._preprocess_function, batched=True
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            eval_accumulation_steps=1,
            include_inputs_for_metrics=False,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
            bf16=torch.cuda.is_available(),  # Enable bf16 if a GPU is available
            load_best_model_at_end=False,
            do_predict=True,
        )

        # Data collator
        if self.gemma.tokenizer is None:
            raise ValueError(
                "Tokenizer is None. Check GemmaTranslationModel initialization."
            )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.gemma.tokenizer, model=self.gemma.model
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.gemma.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,  # Add compute_metrics here
        )

        # Start training
        logger.info("Training...")
        trainer.train()

        # Save the final adapter
        adapter_output_dir = Path(output_dir) / "final_adapter"
        trainer.save_model(str(adapter_output_dir))
        logger.info(f"LoRA adapter saved to {adapter_output_dir}")

        # Log final results
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        results_file = Path(output_dir) / "final_evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Final results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Gemma model using LoRA.")
    parser.add_argument(
        "--model", choices=["270m", "1b"], default="1b", help="Model size to train."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="Evaluation batch size."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate."
    )
    parser.add_argument(
        "--output_dir",
        default=str(paths.results_dir / "lora_finetuning"),
        help="Base output directory for model and results.",
    )
    parser.add_argument(
        "--language_groups",
        nargs="+",
        default=["az_tr_en", "be_uk_en"],
        help="Language groups to use for training (e.g., az_tr_en be_uk_en).",
    )
    args = parser.parse_args()

    # Load .env file
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")

    model_mapping = {"270m": "google/gemma-3-270m-it", "1b": "google/gemma-3-1b-it"}
    model_name = model_mapping[args.model]

    # Create output directory
    finetuning_type = "lora"
    output_dir = Path(args.output_dir) / f"{finetuning_type}_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run trainer
    finetuner = LoRAFinetuner(
        model_name=model_name,
        language_groups=args.language_groups,
        token=token,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project,
    )

    try:
        finetuner.train(
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        logger.info(f"{finetuning_type.upper()} fine-tuning completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
