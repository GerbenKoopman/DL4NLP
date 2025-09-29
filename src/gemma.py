"""
Gemma Translation Model Wrapper
Handles translation tasks using Gemma-3-1B-IT with proper prompting
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaTranslationModel:
    """Gemma-3-1B-IT wrapper for translation tasks"""

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        device: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.token = token
        # Optimize for Mac M4 - prefer MPS, fallback to CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.model = None
        self.tokenizer = None

        # Language codes for prompting (extended for research)
        self.lang_codes = {
            "en": "English",
            "az": "Azerbaijani",
            "be": "Belarusian",
            "tr": "Turkish",
            "uk": "Ukrainian",
            "ru": "Russian",
        }

        # Reverse mapping for validation
        self.valid_lang_codes = {"en", "az", "be", "tr", "uk", "ru"}

    def load_model(self):
        """Load Gemma model and tokenizer with authentication"""
        try:
            # Determine token
            token = self.token or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if token:
                logger.info("Using Hugging Face token for authentication.")
            else:
                logger.warning(
                    "No HUGGINGFACE_HUB_TOKEN found. Model loading may fail if the model is private."
                )

            logger.info(f"Loading {self.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)

            # Optimize dtype and loading for different devices and Gemma 3
            if self.device == "mps":
                # MPS-optimized loading for Gemma 3
                model_dtype = torch.float16
                device_map = None
            elif self.device == "cuda":
                model_dtype = torch.float16
                device_map = "auto"
            else:
                model_dtype = torch.float32
                device_map = None

            # Load model with proper configuration for Gemma 3
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                dtype=model_dtype,
                attn_implementation="eager",  # Use eager attention to avoid issues
                low_cpu_mem_usage=True,
                token=token,
                trust_remote_code=True,
            )

            # Set model to eval mode
            self.model.eval()

            # Configure tokenizer properly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            logger.info("Gemma model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            return False

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        few_shot_examples: Optional[List[Dict]] = None,
        max_length: int = 128,
    ) -> str:
        """Translate text using the Gemma model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Validate language codes
        if source_lang not in self.valid_lang_codes:
            raise ValueError(f"Invalid source language: {source_lang}")
        if target_lang not in self.valid_lang_codes:
            raise ValueError(f"Invalid target language: {target_lang}")

        # Create a simple, direct prompt
        prompt = self._create_prompt(text, source_lang, target_lang, few_shot_examples)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,  # Use greedy decoding for consistency
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode and clean the translation
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = self._clean_translation(full_output, prompt, target_lang)
            return translation

        except Exception as e:
            logger.error(f"Translation failed for text '{text[:50]}...': {e}")
            return ""

    def _create_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """Creates a translation prompt for the model."""
        src_lang_name = self.lang_codes[source_lang]
        tgt_lang_name = self.lang_codes[target_lang]

        if few_shot_examples:
            # TODO: this looks like in-context learning instead?
            instruction = f"Translate the following text from {src_lang_name} to {tgt_lang_name}.\n\n"
            for example in few_shot_examples:
                if (
                    example.get("source_lang") == source_lang
                    and example.get("target_lang") == target_lang
                ):
                    instruction += f"{example['source']} → {example['target']}\n"
            instruction += f"\nNow translate: {text}\nTranslation:"
        else:
            instruction = (
                f"Translate the following text from {src_lang_name} to {tgt_lang_name}. "
                f"Provide only the translated text, without any additional commentary or prefixes.\n\n"
                f'Text: "{text}"\n'
                f"Translation:"
            )

        return instruction

    def _clean_translation(
        self, full_output: str, prompt: str, target_lang: str
    ) -> str:
        """Clean the model's output to extract only the translation."""
        # Remove the prompt from the output
        if full_output.startswith(prompt):
            translation = full_output[len(prompt) :].strip()
        else:
            # Fallback if the prompt is not exactly at the beginning
            # Find the last occurrence of "Translation:" and take the text after it
            parts = full_output.rsplit("Translation:", 1)
            if len(parts) > 1:
                translation = parts[1].strip()
            else:
                translation = full_output.strip()

        # Take the first line, as the model might add extra text
        translation = translation.split("\n")[0].strip()

        # Remove common conversational prefixes and language names
        tgt_lang_name = self.lang_codes.get(target_lang, "")
        prefixes_to_remove = [
            "The translation is",
            "Here is the translation",
            "Here's the translation",
            "Translation",
            f"In {tgt_lang_name}",
            f"{tgt_lang_name} translation",
        ]
        for prefix in prefixes_to_remove:
            # Case-insensitive removal
            if translation.lower().startswith(prefix.lower()):
                translation = translation[len(prefix) :].strip()
                # Remove leading colon if it exists
                if translation.startswith(":"):
                    translation = translation[1:].strip()

        # Remove surrounding quotes or asterisks
        translation = translation.strip().strip("\"'*")

        return translation

    def batch_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> List[str]:
        """Translate multiple texts in batch"""
        translations = []

        for text in texts:
            try:
                translation = self.translate(
                    text, source_lang, target_lang, few_shot_examples
                )
                translations.append(translation)
            except Exception as e:
                logger.warning(f"Translation failed for text '{text[:50]}...': {e}")
                translations.append("")  # Empty string for failed translations

        return translations

    def evaluate_translation_quality(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> Dict[str, float]:
        """Evaluate translation quality using BLEU score"""
        try:
            from sacrebleu import BLEU

            # Generate translations
            translations = self.batch_translate(source_texts, source_lang, target_lang)

            # Calculate BLEU score
            bleu = BLEU()
            score = bleu.corpus_score(translations, [target_texts])

            return {
                "bleu_score": score.score,
                "translation_count": len(translations),
                "success_rate": sum(1 for t in translations if t.strip())
                / len(translations),
            }

        except ImportError:
            logger.warning("sacrebleu not installed. Cannot calculate BLEU score.")
            return {"bleu_score": 0.0, "translation_count": 0, "success_rate": 0.0}


def main():
    """Test the Gemma translation model"""
    # Initialize model
    model = GemmaTranslationModel()
    model.load_model()

    # Test translation
    test_text = "Hello, how are you today?"
    source_lang = "en"
    target_lang = "az"

    print(f"Testing translation:")
    print(f"Source ({source_lang}): {test_text}")

    translation = model.translate(test_text, source_lang, target_lang)
    print(f"Translation ({target_lang}): {translation}")

    # Test few-shot translation
    few_shot_examples = [
        {
            "source": "Good morning!",
            "target": "Sabahınız xeyir!",
            "source_lang": "en",
            "target_lang": "az",
        }
    ]

    print(f"\nTesting few-shot translation:")
    few_shot_translation = model.translate(
        test_text, source_lang, target_lang, few_shot_examples
    )
    print(f"Few-shot translation: {few_shot_translation}")


if __name__ == "__main__":
    main()
