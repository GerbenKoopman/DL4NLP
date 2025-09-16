"""
Simple TED Talk Data Cleaner
Removes __NULL__ entries and creates task dictionaries for meta-learning
"""

import pandas as pd
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TEDDataCleaner:
    """Clean and prepare TED talk data for meta-learning tasks"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        # Target languages for meta-learning
        self.base_langs = ['az', 'be', 'en']  # Azerbaijani, Belarusian, English
        self.target_langs = ['tr', 'uk']      # Turkish, Ukrainian (for evaluation)
        
    def clean_dataset(self, split_name: str) -> pd.DataFrame:
        """Clean a single dataset split"""
        file_path = self.data_dir / f"all_talks_{split_name}.tsv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading {split_name} split from {file_path}")
        df = pd.read_csv(file_path, sep='\t')
        
        original_size = len(df)
        logger.info(f"Original size: {original_size} rows")
        
        # Get available language columns
        available_langs = [col for col in df.columns if col in (self.base_langs + self.target_langs)]
        logger.info(f"Available languages: {available_langs}")
        
        # Remove rows where any target language has __NULL__ or empty values
        for lang in available_langs:
            if lang in df.columns:
                # Remove __NULL__ and empty entries
                df = df[~df[lang].isin(['__NULL__', '', None])]
                df = df[df[lang].notna()]
                df = df[df[lang].str.strip() != '']
        
        cleaned_size = len(df)
        logger.info(f"Cleaned size: {cleaned_size} rows ({original_size - cleaned_size} removed)")
        
        return df[['talk_name'] + available_langs]
    
    def create_task_dictionaries(self, df: pd.DataFrame, split_name: str) -> List[Dict]:
        """Create task dictionaries for meta-learning"""
        tasks = []
        
        available_langs = [col for col in df.columns if col != 'talk_name']
        
        # Create all language pair combinations within base languages
        for src_lang in available_langs:
            for tgt_lang in available_langs:
                if src_lang != tgt_lang:
                    task_type = f"{src_lang}_{tgt_lang}"
                    
                    # Only create tasks for base languages during training
                    if split_name == 'train':
                        if src_lang not in self.base_langs or tgt_lang not in self.base_langs:
                            continue
                    
                    for _, row in df.iterrows():
                        src_text = str(row[src_lang]).strip()
                        tgt_text = str(row[tgt_lang]).strip()
                        
                        if src_text and tgt_text and src_text != 'nan' and tgt_text != 'nan':
                            task_dict = {
                                'talk_name': row['talk_name'],
                                'source_text': src_text,
                                'target_text': tgt_text,
                                'source_lang': src_lang,
                                'target_lang': tgt_lang,
                                'task_type': task_type,
                                'split': split_name
                            }
                            tasks.append(task_dict)
        
        logger.info(f"Created {len(tasks)} task instances for {split_name}")
        return tasks
    
    def save_tasks(self, tasks: List[Dict], output_path: str):
        """Save tasks as pickle file"""
        with open(output_path, 'wb') as f:
            pickle.dump(tasks, f)
        logger.info(f"Saved {len(tasks)} tasks to {output_path}")
    
    def process_all_splits(self):
        """Process all dataset splits"""
        all_tasks = {}
        
        for split in ['train', 'dev', 'test']:
            try:
                # Clean the dataset
                df = self.clean_dataset(split)
                
                # Create task dictionaries
                tasks = self.create_task_dictionaries(df, split)
                all_tasks[split] = tasks
                
                # Save individual split
                output_path = self.data_dir / f"{split}_tasks.pkl"
                self.save_tasks(tasks, output_path)
                
            except FileNotFoundError as e:
                logger.warning(f"Skipping {split}: {e}")
        
        # Save combined tasks
        output_path = self.data_dir / "all_tasks.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(all_tasks, f)
        
        logger.info("Completed processing all splits")
        return all_tasks

def main():
    parser = argparse.ArgumentParser(description="Clean TED talk data")
    parser.add_argument("--data_dir", default="datasets", help="Data directory")
    parser.add_argument("--split", choices=['train', 'dev', 'test', 'all'], 
                       default='all', help="Which split to process")
    
    args = parser.parse_args()
    
    cleaner = TEDDataCleaner(args.data_dir)
    
    if args.split == 'all':
        cleaner.process_all_splits()
    else:
        df = cleaner.clean_dataset(args.split)
        tasks = cleaner.create_task_dictionaries(df, args.split)
        output_path = Path(args.data_dir) / f"{args.split}_tasks.pkl"
        cleaner.save_tasks(tasks, output_path)

if __name__ == "__main__":
    main()
