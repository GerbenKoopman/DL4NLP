# Deep Learning for Natural Language Processing

Repository for course mini-project.

## Multilingual Neural Machine Translation for Low-Resource Languages

### Project Description

We investigate whether LoRA fine-tuning on Azerbaijani-English and Belarusian-English translation pairs can enable improved translation quality on closely related languages (Turkish, Ukrainian) with minimal parallel data. Specifically, can parameter-efficient fine-tuning on English-bridged base pairs improve cross-lingual transfer within language families?

### Hypothesis

Our hypothesis is that LoRA fine-tuning on base language pairs (with English) internalizes morphological, syntactic, and subword regularities that transfer more effectively to closely related languages within the same family (Azerbaijani ↔ Turkish, Belarusian ↔ Ukrainian) compared to unrelated language pairs.

### Dataset

We use the [TED Talks multilingual corpus](http://aclweb.org/anthology/N18-2084), downloaded from <http://phontron.com/data/ted_talks.tar.gz>, containing translated sentences in 60 languages. This provides aligned sentence-level translations for our transfer learning experiments.

### Technologies

We use **LoRA (Low-Rank Adaptation)** for efficient fine-tuning of Gemma-3 models:

- **Models**: Gemma-3 variants - 270M and 1B instruction-tuned (`google/gemma-3-*-it`)
- **Fine-tuning**: LoRA adapters with parameter-efficient updates
- **Monitoring**: Weights & Biases (wandb) integration for training metrics (BLEU, chrF)
- **Evaluation**: Comprehensive evaluation suite (zero-shot baseline vs LoRA fine-tuned)
- **Transfer Learning**: Training on base pairs (az↔en, be↔en) and evaluating on target pairs (az↔tr, be↔uk)

> **Note**: 1B model is recommended for development; 270M for quick experiments.

## Setup & Environment

1. **Create environment:**

```bash
conda env create -f environment.yml
conda activate dl4nlp
```

2. **Set up environment variables** (create `.env` file in project root):

```bash
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=lora-finetuning
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

## Quick Start Guide

### Basic Workflow

```bash
# 1. Navigate to src directory
cd src

# 2. Prepare data (if not already done)
python clean_data.py --split all

# 3. Run baseline evaluation (zero-shot)
python baseline_eval.py --model 1b --eval_pair_policy target

# 4. Train LoRA fine-tuning
python train_lora.py --model 1b --epochs 3 --train_pair_policy base --eval_pair_policy target

# 5. Evaluate trained model (automatic after training)
# Results saved to results/lora_base_to_target_1b_5e/lora_1b/
```

### Advanced Training Options

```bash
# Train with specific language groups and parameters
python train_lora.py \
  --model 1b \
  --epochs 5 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 4 \
  --train_pair_policy base \
  --eval_pair_policy target \
  --language_groups az_tr_en be_uk_en \
  --output_dir results/my_experiment

# Train 270M model for quick experiments
python train_lora.py \
  --model 270m \
  --epochs 3 \
  --max_train_samples 100 \
  --max_eval_samples 50

# Baseline evaluation with different policies
python baseline_eval.py \
  --model 1b \
  --eval_pair_policy all \
  --max_eval_samples 100
```

## Available Scripts

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| `train_lora.py` | Train LoRA fine-tuning | `--model`, `--epochs`, `--train_pair_policy`, `--eval_pair_policy` |
| `baseline_eval.py` | Run zero-shot baseline evaluation | `--model`, `--eval_pair_policy`, `--max_eval_samples` |
| `clean_data.py` | Process TED Talks data | `--split` (train/dev/test/all) |
| `gemma.py` | Gemma model wrapper and translation | `--model`, `--use_lora` |
| `evaluation.py` | Translation evaluation metrics | BLEU, chrF computation |

## Key Features

✅ **Implemented:**

- LoRA fine-tuning with parameter-efficient updates
- Transfer learning from base pairs (az↔en, be↔en) to target pairs (az↔tr, be↔uk)
- Weights & Biases integration with BLEU/chrF metrics logging  
- Comprehensive evaluation suite (zero-shot baseline vs LoRA fine-tuned)
- Flexible pair policies (base, target, all)
- Multi-scale support (270M, 1B models)
- Reproducible experiments with command-line configuration

## Experimental Setup

### Language Triads

- **Turkic family**: `az_tr_en` (Azerbaijani–Turkish–English)
- **Slavic family**: `be_uk_en` (Belarusian–Ukrainian–English)

### Training Strategy

- **Base pairs**: az↔en, be↔en, tr↔en, uk↔en (pairs involving English)
- **Target pairs**: az↔tr, be↔uk (within-family pairs)
- **Transfer protocol**: Train on base pairs, evaluate on target pairs

### Evaluation Metrics

- **BLEU**: Word-level n-gram precision with brevity penalty
- **chrF**: Character-level F-scores for morphological variation

## Results

Our experiments show substantial improvements:

- **Baseline (zero-shot)**: BLEU ~0.87, chrF ~9.63
- **LoRA fine-tuned**: BLEU ~44.12, chrF ~59.54
- **Improvement**: 37x better BLEU, demonstrating effective cross-lingual transfer

## Project Structure

```text
DL4NLP/
├── src/                       # Source code
│   ├── train_lora.py          # Main LoRA training script
│   ├── baseline_eval.py       # Zero-shot baseline evaluation
│   ├── clean_data.py          # Data processing
│   ├── gemma.py               # Gemma model wrapper
│   ├── evaluation.py          # Evaluation metrics
│   └── paths.py               # Path configuration
├── datasets/                  # Processed data files
│   ├── az_tr_en_*.pkl         # Turkic family data
│   ├── be_uk_en_*.pkl         # Slavic family data
│   └── all_talks_*.tsv        # Raw TED Talks data
├── results/                   # Training and evaluation results
│   ├── baseline_1b/           # Baseline evaluation results
│   ├── lora_base_to_target_1b_5e/  # LoRA training results
│   └── comparison_1b/         # Comparison results
├── environment.yml            # Conda environment specification
└── README.md                  # This file
```

## Command Line Arguments

### train_lora.py

- `--model`: Model size (270m, 1b)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--train_pair_policy`: Training pairs (base, target, all)
- `--eval_pair_policy`: Evaluation pairs (base, target, all)
- `--language_groups`: Language groups to use
- `--max_train_samples`: Limit training examples
- `--max_eval_samples`: Limit evaluation examples

### baseline_eval.py

- `--model`: Model size (270m, 1b)
- `--eval_pair_policy`: Evaluation pairs (base, target, all)
- `--max_eval_samples`: Limit evaluation examples
- `--output_dir`: Output directory for results

## Reproducibility

All experiments are fully reproducible with:

- Fixed random seeds
- Command-line parameter specification
- Saved LoRA adapters and tokenizer states
- W&B logging for training metrics
- Consistent evaluation protocols

## Future Work

- Few-shot evaluation with in-context examples
- Cross-family transfer experiments (az↔uk, be↔tr)
- Ablation studies on LoRA rank and learning rates
- Comparison with other parameter-efficient methods
- Scaling to larger models (4B, 7B)
