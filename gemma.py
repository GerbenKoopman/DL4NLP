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

            # Move to device if not using device_map
            if device_map is None:
                self.model = self.model.to(self.device)

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

    def create_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """Create translation prompt compatible with Gemma 3 chat template"""

        # Validate language codes
        if source_lang not in self.valid_lang_codes:
            raise ValueError(f"Invalid source language: {source_lang}")
        if target_lang not in self.valid_lang_codes:
            raise ValueError(f"Invalid target language: {target_lang}")

        # Create a simple, direct prompt that should work better with Gemma 3
        if few_shot_examples:
            # With examples
            instruction = f"Translate the following text from {self.lang_codes[source_lang]} to {self.lang_codes[target_lang]}.\n\n"
            instruction += "Examples:\n"
            for example in few_shot_examples:
                if (
                    example["source_lang"] == source_lang
                    and example["target_lang"] == target_lang
                ):
                    instruction += f"{example['source']} → {example['target']}\n"
            instruction += f"\nNow translate: {text}\nTranslation:"
        else:
            # Zero-shot - very simple format
            instruction = f"Translate this {self.lang_codes[source_lang]} text to {self.lang_codes[target_lang]} and only the translation: {text}. DO NOT RETURN ANYTHING ELSE LIKE EXAMPLES OR ANYTHING ELSE APART FROM THE TRANSLATION."

        # Use the simple instruction directly for now
        # Gemma 3 might work better without complex chat templates for translation
        return instruction

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        few_shot_examples: Optional[List[Dict]] = None,
        max_length: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Translate text using either chat template or direct completion"""

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Use direct completion with better constraints
        return self._translate_direct(
            text, source_lang, target_lang, few_shot_examples, max_length, temperature
        )

    def _translate_direct(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        few_shot_examples: Optional[List[Dict]] = None,
        max_length: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Direct completion approach for smaller models"""

        # Create instruction that explicitly asks for only translation
        if few_shot_examples:
            instruction = f"Translate the following text from {self.lang_codes[source_lang]} to {self.lang_codes[target_lang]}.\n\n"
            instruction += "Examples:\n"
            for example in few_shot_examples:
                if (
                    example["source_lang"] == source_lang
                    and example["target_lang"] == target_lang
                ):
                    instruction += f"{example['source']} → {example['target']}\n"
            instruction += f"\nNow translate: {text}\nTranslation:"
        else:
            # Zero-shot - use a more direct format that works better with instruction models
            instruction = f"Translate '{text}' from {self.lang_codes[source_lang]} to {self.lang_codes[target_lang]}:"

        prompt = instruction

        try:
            # Tokenize directly
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with strict constraints
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(
                        max_length, 20
                    ),  # Very short to force single word/phrase
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                )

            # Extract new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
            translation = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            # Aggressive cleaning to get just the translation
            translation = translation.split("\n")[0].strip()
            translation = translation.split(".")[0].strip()  # Remove sentences

            # Remove common prefixes that indicate explanations
            prefixes_to_remove = [
                "The most common translation",
                "There are several ways",
                "The translation",
                "In Turkish",
                "The word",
                "This means",
                "It means",
                "You can say",
                "The answer is",
                "The correct translation",
            ]

            for prefix in prefixes_to_remove:
                if translation.lower().startswith(prefix.lower()):
                    # Find the first quote or colon after the prefix
                    colon_pos = translation.find(":")
                    quote_pos = translation.find('"')
                    if colon_pos > 0 and quote_pos > 0:
                        translation = translation[
                            min(colon_pos, quote_pos) + 1 :
                        ].strip()
                    elif colon_pos > 0:
                        translation = translation[colon_pos + 1 :].strip()
                    elif quote_pos > 0:
                        translation = translation[quote_pos + 1 :].strip()
                    break

            # Remove markdown formatting
            translation = (
                translation.split("**")[1] if "**" in translation else translation
            )
            translation = (
                translation.split("*")[1]
                if "*" in translation and not translation.startswith("*")
                else translation
            )
            translation = translation.strip("\"'")  # Remove quotes
            translation = translation.split("(")[
                0
            ].strip()  # Remove parenthetical explanations
            translation = translation.split('"')[
                0
            ].strip()  # Remove any remaining quotes

            return translation

        except Exception as e:
            logger.error(f"Direct translation failed: {e}")
            return ""

    def _translate_chat(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        few_shot_examples: Optional[List[Dict]] = None,
        max_length: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Chat template approach for larger models"""

        # Zero-shot: strict, direct translation instruction
        src_lang_name = self.lang_codes.get(source_lang, source_lang.upper())
        tgt_lang_name = self.lang_codes.get(target_lang, target_lang.upper())

        instruction = f"Translate this word to {tgt_lang_name}: {text}"

        messages = [{"role": "user", "content": instruction}]

        try:
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )

            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with proper parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Extract only the new generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
            translation = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            logger.debug(f"Generated translation: '{translation}'")
            return translation

        except Exception as e:
            logger.error(f"Chat translation failed: {e}")
            return ""

    def _clean_translation(self, translation: str, target_lang: str) -> str:
        """Clean and validate translation output"""
        translation = translation.strip()

        # Remove parenthetical annotations like (Formal), (Informal), etc.
        import re

        translation = re.sub(r"\s*\([^)]*\)\s*", "", translation)
        translation = re.sub(r"\s*\[[^\]]*\]\s*", "", translation)

        # Remove common verbose prefixes
        prefixes_to_remove = [
            "Here's a translation",
            "Here's the translation",
            "Translation:",
            "The translation is:",
            f"{self.lang_codes.get(target_lang, '')}:",
            "aiming for accuracy and natural flow:",
            "**",
            "*",
        ]

        for prefix in prefixes_to_remove:
            if translation.lower().startswith(prefix.lower()):
                translation = translation[len(prefix) :].strip()

        # Remove markdown formatting and extra structure
        translation = translation.replace("**", "").replace("*", "")

        # Take only the first line/sentence (before newlines or colons)
        translation = translation.split("\n")[0].split(":")[-1].strip()

        # Remove quotes if the entire translation is quoted
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1].strip()

        # Remove common suffixes
        suffixes_to_remove = ["!", ".", "?"]
        for suffix in suffixes_to_remove:
            if translation.endswith(suffix) and len(translation) > 1:
                # Only remove if there are other punctuation marks or it's clearly an annotation
                continue  # Keep natural punctuation

        return translation.strip()

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
