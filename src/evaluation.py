"""
Evaluation Metrics for Low-Resource Neural Machine Translation
Implements sacreBLEU and chrF metrics for translation quality assessment
"""

import re
import logging
from typing import List, Dict
from collections import Counter
import math

logger = logging.getLogger(__name__)

class TranslationEvaluator:
    """Translation evaluation using sacreBLEU and chrF metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_translation(self, translation: str, reference: str, 
                           metrics: List[str] = ['bleu', 'chrf']) -> Dict[str, float]:
        """Evaluate translation quality using specified metrics"""
        results = {}
        
        if not translation.strip() or not reference.strip():
            return {metric: 0.0 for metric in metrics}
        
        for metric in metrics:
            if metric == 'bleu':
                results[metric] = self.compute_sacrebleu(translation, reference)
            elif metric == 'chrf':
                results[metric] = self.compute_chrf(translation, reference)
            else:
                logger.warning(f"Unknown metric: {metric}. Supported: bleu, chrf")
                results[metric] = 0.0
        
        return results
    
    def compute_sacrebleu(self, translation: str, reference: str, max_n: int = 4) -> float:
        """
        Compute sacreBLEU score (simplified implementation following sacrebleu principles)
        Uses standard tokenization and smoothing for low-resource languages
        """
        if not translation.strip() or not reference.strip():
            return 0.0
        
        # Tokenize (simple whitespace + punctuation separation)
        trans_tokens = self._tokenize(translation)
        ref_tokens = self._tokenize(reference)
        
        if not trans_tokens or not ref_tokens:
            return 0.0
        
        # Brevity penalty
        trans_len = len(trans_tokens)
        ref_len = len(ref_tokens)
        
        if trans_len == 0:
            return 0.0
        
        # Standard sacreBLEU brevity penalty
        if trans_len < ref_len:
            bp = math.exp(1 - ref_len / trans_len)
        else:
            bp = 1.0
        
        # N-gram precisions with smoothing
        precisions = []
        for n in range(1, min(max_n + 1, trans_len + 1)):
            trans_ngrams = self._get_ngrams(trans_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            if not trans_ngrams:
                precisions.append(0.0)
                continue
            
            matches = 0
            for ngram in trans_ngrams:
                if ngram in ref_ngrams:
                    matches += min(trans_ngrams[ngram], ref_ngrams[ngram])
            
            # Add smoothing for low-resource scenarios (epsilon smoothing)
            precision = (matches + 1e-7) / (sum(trans_ngrams.values()) + 1e-7)
            precisions.append(precision)
        
        if not precisions or all(p == 0 for p in precisions):
            return 0.0
        
        # Geometric mean of precisions
        log_sum = sum(math.log(max(p, 1e-10)) for p in precisions)
        geo_mean = math.exp(log_sum / len(precisions))
        
        return bp * geo_mean * 100  # Scale to 0-100 range
    
    def compute_chrf(self, translation: str, reference: str, 
                     n_char: int = 6, n_word: int = 2, beta: float = 2.0) -> float:
        """
        Compute chrF score (character-level F-score)
        Well-suited for low-resource languages and morphologically rich languages
        """
        if not translation.strip() or not reference.strip():
            return 0.0
        
        # Character-level n-grams
        char_precision, char_recall = self._chrf_precision_recall(
            translation, reference, n_char, level='char'
        )
        
        # Word-level n-grams
        word_precision, word_recall = self._chrf_precision_recall(
            translation, reference, n_word, level='word'
        )
        
        # Combine character and word level scores
        avg_precision = (char_precision + word_precision) / 2
        avg_recall = (char_recall + word_recall) / 2
        
        # F-beta score (beta=2 gives more weight to recall)
        if avg_precision + avg_recall == 0:
            return 0.0
        
        f_score = (1 + beta**2) * avg_precision * avg_recall / (
            beta**2 * avg_precision + avg_recall
        )
        
        return f_score * 100  # Scale to 0-100 range
    
    def _chrf_precision_recall(self, translation: str, reference: str, 
                              max_n: int, level: str = 'char'):
        """Calculate precision and recall for chrF"""
        if level == 'char':
            trans_items = list(translation.replace(' ', ''))
            ref_items = list(reference.replace(' ', ''))
        else:  # word level
            trans_items = translation.split()
            ref_items = reference.split()
        
        if not trans_items or not ref_items:
            return 0.0, 0.0
        
        total_precision = 0.0
        total_recall = 0.0
        total_n = 0
        
        for n in range(1, min(max_n + 1, len(trans_items) + 1, len(ref_items) + 1)):
            trans_ngrams = self._get_ngrams(trans_items, n)
            ref_ngrams = self._get_ngrams(ref_items, n)
            
            if not trans_ngrams or not ref_ngrams:
                continue
            
            matches = 0
            for ngram in trans_ngrams:
                if ngram in ref_ngrams:
                    matches += min(trans_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(trans_ngrams.values()) if trans_ngrams else 0.0
            recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0.0
            
            total_precision += precision
            total_recall += recall
            total_n += 1
        
        avg_precision = total_precision / total_n if total_n > 0 else 0.0
        avg_recall = total_recall / total_n if total_n > 0 else 0.0
        
        return avg_precision, avg_recall
    
    def batch_evaluate(self, translations: List[str], references: List[str], 
                      metrics: List[str] = ['bleu', 'chrf']) -> Dict[str, float]:
        """Evaluate multiple translation pairs and return average scores"""
        if len(translations) != len(references):
            raise ValueError("Number of translations and references must match")
        
        if not translations:
            return {metric: 0.0 for metric in metrics}
        
        # Collect all scores
        all_scores = {metric: [] for metric in metrics}
        
        for trans, ref in zip(translations, references):
            scores = self.evaluate_translation(trans, ref, metrics)
            for metric in metrics:
                all_scores[metric].append(scores[metric])
        
        # Compute averages
        avg_scores = {}
        for metric in metrics:
            avg_scores[metric] = sum(all_scores[metric]) / len(all_scores[metric])
        
        return avg_scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for sacreBLEU compatibility"""
        # Basic tokenization - separate punctuation and normalize
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        tokens = text.lower().strip().split()
        return [token for token in tokens if token]
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)

def main():
    """Test the evaluation metrics with low-resource language examples"""
    evaluator = TranslationEvaluator()
    
    # Test cases including some low-resource language scenarios
    test_cases = [
        ("Hello world", "Hello world", "Perfect match"),
        ("Good morning everyone", "Good morning everybody", "Close match"),
        ("The cat sat on the mat", "A cat was sitting on the mat", "Paraphrase"),
        ("", "Hello", "Empty translation"),
        ("Salam dünya", "Hello world", "Different script (Az-En)"),
        ("Добры раніца", "Good morning", "Cyrillic script (Be-En)")
    ]
    
    print("🧪 TESTING TRANSLATION EVALUATION METRICS")
    print("=" * 60)
    
    for translation, reference, description in test_cases:
        print(f"\n📝 {description}")
        print(f"Translation: '{translation}'")
        print(f"Reference:   '{reference}'")
        
        scores = evaluator.evaluate_translation(translation, reference, ['bleu', 'chrf'])
        
        print("Scores:")
        for metric, score in scores.items():
            print(f"  {metric.upper()}: {score:.2f}")

if __name__ == "__main__":
    main()