"""
Baseline Evaluation Script
Zero-shot evaluation of Gemma models on translation tasks
"""

import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from gemma import GemmaTranslationModel
from evaluation import TranslationEvaluator
from cache import get_cached_gemma_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineEvaluator:
    """Evaluate Gemma models on zero-shot translation tasks"""
    
    def __init__(self, model_name: str, data_dir: str = "datasets"):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.evaluator = TranslationEvaluator()
        self.model = None
        
        # Target language pairs for evaluation
        self.base_pairs = [
            ('az', 'en'), ('en', 'az'),  # Azerbaijani ↔ English
            ('be', 'en'), ('en', 'be'),  # Belarusian ↔ English
        ]
        
        self.target_pairs = [
            ('tr', 'en'), ('en', 'tr'),  # Turkish ↔ English (transfer evaluation)
            ('uk', 'en'), ('en', 'uk'),  # Ukrainian ↔ English (transfer evaluation)
        ]
    
    def load_model(self):
        """Load the Gemma model"""
        logger.info(f"Loading {self.model_name}...")
        self.model = get_cached_gemma_model(self.model_name)
        logger.info("Model loaded successfully")
    
    def load_test_data(self) -> Dict[str, List[Dict]]:
        """Load test data for evaluation"""
        test_data = {}
        
        # Try to load processed test tasks
        test_file = self.data_dir / "test_tasks.pkl"
        if test_file.exists():
            with open(test_file, 'rb') as f:
                test_tasks = pickle.load(f)
            
            # Group by task type
            for task in test_tasks:
                task_type = task['task_type']
                if task_type not in test_data:
                    test_data[task_type] = []
                test_data[task_type].append(task)
            
            logger.info(f"Loaded test data with {len(test_data)} task types")
        else:
            logger.warning(f"Test data file not found: {test_file}")
            logger.info("Run data/ted_clean.py first to process the data")
        
        return test_data
    
    def evaluate_language_pair(self, source_lang: str, target_lang: str, 
                             test_examples: List[Dict], max_examples: int = 50) -> Dict[str, float]:
        """Evaluate zero-shot performance on a language pair"""
        logger.info(f"Evaluating {source_lang} → {target_lang} (zero-shot)")
        
        if not test_examples:
            logger.warning(f"No test examples for {source_lang} → {target_lang}")
            return {'bleu': 0.0, 'chrf': 0.0, 'count': 0}
        
        # Limit examples for efficiency
        examples = test_examples[:max_examples]
        translations = []
        references = []
        
        for i, example in enumerate(examples):
            try:
                # Generate zero-shot translation with greedy decoding
                translation = self.model.translate(
                    example['source_text'],
                    source_lang,
                    target_lang,
                    few_shot_examples=None,  # Zero-shot
                    max_length=32,  # Very short for testing
                    temperature=0.0  # Greedy decoding for stability
                )
                
                translations.append(translation)
                references.append(example['target_text'])
                
                # Debug: Print first few examples
                if i < 3:
                    logger.info(f"Example {i+1}:")
                    logger.info(f"  Source: {example['source_text'][:100]}...")
                    logger.info(f"  Translation: {translation[:100]}...")
                    logger.info(f"  Reference: {example['target_text'][:100]}...")
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(examples)} examples")
                
            except Exception as e:
                logger.error(f"Translation failed for example {i}: {e}")
                translations.append("")  # Empty translation for failed cases
                references.append(example['target_text'])
        
        # Evaluate using batch metrics
        scores = self.evaluator.batch_evaluate(translations, references, ['bleu', 'chrf'])
        scores['count'] = len(examples)
        
        # Debug: Show some stats
        non_empty_translations = [t for t in translations if t.strip()]
        logger.info(f"Translation stats: {len(non_empty_translations)}/{len(translations)} non-empty")
        
        logger.info(f"Results for {source_lang} → {target_lang}: "
                   f"BLEU={scores['bleu']:.2f}, chrF={scores['chrf']:.2f}")
        
        return scores
    
    def run_baseline_evaluation(self, output_file: str = None, max_examples: int = 50) -> Dict[str, Dict[str, float]]:
        """Run complete baseline evaluation"""
        if self.model is None:
            self.load_model()
        
        # Load test data
        test_data = self.load_test_data()
        
        if not test_data:
            logger.error("No test data available")
            return {}
        
        results = {
            'model_name': self.model_name,
            'evaluation_type': 'zero_shot_baseline',
            'results': {}
        }
        
        # Evaluate all language pairs
        all_pairs = self.base_pairs + self.target_pairs
        
        for source_lang, target_lang in all_pairs:
            task_type = f"{source_lang}_{target_lang}"
            
            if task_type in test_data:
                scores = self.evaluate_language_pair(
                    source_lang, target_lang, test_data[task_type], max_examples
                )
                results['results'][task_type] = scores
            else:
                logger.warning(f"No test data for {task_type}")
                results['results'][task_type] = {'bleu': 0.0, 'chrf': 0.0, 'count': 0}
        
        # Calculate averages
        self._calculate_averages(results)
        
        # Save results
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _calculate_averages(self, results: Dict):
        """Calculate average scores across language pairs"""
        base_scores = {'bleu': [], 'chrf': []}
        target_scores = {'bleu': [], 'chrf': []}
        
        for source_lang, target_lang in self.base_pairs:
            task_type = f"{source_lang}_{target_lang}"
            if task_type in results['results']:
                scores = results['results'][task_type]
                if scores['count'] > 0:
                    base_scores['bleu'].append(scores['bleu'])
                    base_scores['chrf'].append(scores['chrf'])
        
        for source_lang, target_lang in self.target_pairs:
            task_type = f"{source_lang}_{target_lang}"
            if task_type in results['results']:
                scores = results['results'][task_type]
                if scores['count'] > 0:
                    target_scores['bleu'].append(scores['bleu'])
                    target_scores['chrf'].append(scores['chrf'])
        
        # Add averages to results
        results['averages'] = {
            'base_languages': {
                'bleu': sum(base_scores['bleu']) / len(base_scores['bleu']) if base_scores['bleu'] else 0.0,
                'chrf': sum(base_scores['chrf']) / len(base_scores['chrf']) if base_scores['chrf'] else 0.0
            },
            'target_languages': {
                'bleu': sum(target_scores['bleu']) / len(target_scores['bleu']) if target_scores['bleu'] else 0.0,
                'chrf': sum(target_scores['chrf']) / len(target_scores['chrf']) if target_scores['chrf'] else 0.0
            }
        }
    
    def _save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print(f"BASELINE EVALUATION SUMMARY - {results['model_name']}")
        print("="*60)
        
        if 'averages' in results:
            print(f"\nBase Languages (Az/Be ↔ En):")
            print(f"  BLEU: {results['averages']['base_languages']['bleu']:.2f}")
            print(f"  chrF: {results['averages']['base_languages']['chrf']:.2f}")
            
            print(f"\nTarget Languages (Tr/Uk ↔ En):")
            print(f"  BLEU: {results['averages']['target_languages']['bleu']:.2f}")
            print(f"  chrF: {results['averages']['target_languages']['chrf']:.2f}")
        
        print(f"\nDetailed Results:")
        for task_type, scores in results['results'].items():
            if scores['count'] > 0:
                print(f"  {task_type}: BLEU={scores['bleu']:.2f}, chrF={scores['chrf']:.2f} ({scores['count']} examples)")

def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation of Gemma models")
    parser.add_argument("--model", choices=["270m", "1b"], default="270m",
                       help="Model size to evaluate")
    parser.add_argument("--data_dir", default="datasets", help="Data directory")
    parser.add_argument("--output", default="results/baseline_results.json",
                       help="Output file for results")
    parser.add_argument("--max_examples", type=int, default=50,
                       help="Maximum examples per language pair")
    
    args = parser.parse_args()
    
    # Map model choices to actual model names
    model_mapping = {
        "270m": "google/gemma-3-270m-it",
        "1b": "google/gemma-3-1b-it"
    }
    
    model_name = model_mapping[args.model]
    
    # Run evaluation
    evaluator = BaselineEvaluator(model_name, args.data_dir)
    results = evaluator.run_baseline_evaluation(args.output, args.max_examples)
    
    if results:
        evaluator.print_summary(results)
    else:
        logger.error("Evaluation failed")

if __name__ == "__main__":
    main()
