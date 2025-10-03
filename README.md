# Deep Learning for Natural Language Processing

Repository for course mini-project.

## Multilingual Neural Machine Translation for Low-Resource Languages

### Project description

We investigate whether meta-learning on Azerbaijani-English and Belarusian-English translation pairs can enable rapid few-shot adaptation to closely related languages (Turkish, Ukrainian) with minimal parallel data. Specifically, can meta-learning with Reptile on pre-chosen language pairs improve few-shot performance on linguistically similar languages?

### Hypothesis

Our hypothesis is that meta-learned representations from the base multilingual NMT model will transfer more effectively to related languages within the same family (Azerbaijani $\to$ Turkish, Belarusian $\to$ Ukrainian) compared to cross-family transfers.

### Dataset

As our dataset we will use the [TED Talks multilingual corpus](http://aclweb.org/anthology/N18-2084), download at <http://phontron.com/data/ted_talks.tar.gz>, containing translated sentences in 60 languages. This will provide the data for the meta-learning task, as well as a held-out validation set.

### Technologies

We use **Reptile meta-learning** with **QLoRA** for efficient fine-tuning of Gemma 3 models:
- **Models**: Primary model Gemma 3 1B (`google/gemma-3-1b-it`)
- **Meta-learning**: Reptile algorithm for few-shot adaptation
- **Fine-tuning**: QLoRA (Quantized Low-Rank Adaptation) with 4-bit quantization
- **Monitoring**: Weights & Biases (wandb) integration for training metrics (BLEU, chrF, meta-average)
- **Evaluation**: Comprehensive evaluation suite (zero-shot, few-shot, transfer)

> **Note**: The code supports both 1B and 270M models, but focusing on 1B based on recent commits and better performance potential.

## Setup & Environment

1. **Create environment:**
```bash
conda env create -f environment.yml
conda activate dl4nlp
```

2. **Set up environment variables** (create `.env` file in project root):
```bash
WANDB_API_KEY=your_wandb_api_key_here
WANDB_ENTITY=gerbennkoopman-university-of-amsterdam
WANDB_PROJECT=reptile-meta-learning
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

## Quick Start Guide

### Basic Workflow:

```bash
# 1. Navigate to src directory
cd src

# 2. Prepare data (if not already done)
python clean_data.py --split all

# 3. Run baseline evaluation 
python baseline_eval.py --model 1b --max_examples 100

# 4. Train Reptile meta-learning with QLoRA
python train_reptile.py --model 1b --meta_steps 50 --inner_steps 5

# 5. Evaluate meta-learned model
python evaluate_reptile.py --model 1b --support_size 5

# 6. Compare with baseline (optional)
python evaluate_reptile.py --model 1b --baseline_file results/baseline_1b.json

# 7. Generate analysis plots
python analyze_results.py
```

### Advanced Training Options:

```bash
# Train with specific adapter mode and language groups
python train_reptile.py \
  --model 1b \
  --meta_steps 100 \
  --inner_steps 5 \
  --support_size 5 \
  --query_size 3 \
  --adapter_mode all \
  --language_groups az_tr_en be_uk_en \
  --output_dir results/reptile_1b
```

### Results analysis options

You can filter what gets plotted:

- Training history filters:
  - Only specific series (task types) and hide meta average:

    ```bash
    python analyze_results.py --train_tasks be_en,en_be --no_meta_average
    ```

  - Include meta average plus selected series:

    ```bash
    python analyze_results.py --train_tasks be_en,en_be,meta_average
    ```

- Evaluation summary filters:
  - Focus on transfer to target languages only (recommended):

    ```bash
    python analyze_results.py --tasks az_tr,be_uk --eval_types transfer_1,transfer_5
    ```

  - Base language few-shot focus:

    ```bash
    python analyze_results.py --tasks az_en,en_az,be_en,en_be --eval_types zero_shot,few_shot_1,few_shot_5
    ```

  - Single task, all evaluation types:

    ```bash
    python analyze_results.py --tasks az_tr
    ```

  - Plot everything (default):

    ```bash
    python analyze_results.py
    ```

## Available Scripts

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| `train_reptile.py` | Train Reptile meta-learning | `--model`, `--meta_steps`, `--adapter_mode`, `--language_groups` |
| `evaluate_reptile.py` | Evaluate trained models | `--model`, `--baseline_file`, `--support_size` |
| `baseline_eval.py` | Run baseline evaluations | `--model`, `--max_examples` |
| `clean_data.py` | Process TED Talks data | `--split` (train/dev/test/all) |
| `analyze_results.py` | Generate plots and analysis | `--train_tasks`, `--eval_types` |

## Key Features

✅ **Implemented:**
- QLoRA fine-tuning with 4-bit quantization for memory efficiency
- Reptile meta-learning algorithm adapted for QLoRA weights
- Weights & Biases integration with BLEU/chrF metrics logging  
- Comprehensive evaluation suite (zero-shot, few-shot, transfer)
- Flexible adapter modes (all languages, az_en, be_en)
- Automated results analysis and visualization

🔄 **In Development:**
- Expanded evaluation metrics and analysis
- Additional meta-learning baselines for comparison

## Project Structure

```
DL4NLP/
├── src/                    # Source code
│   ├── train_reptile.py    # Main training script
│   ├── evaluate_reptile.py # Evaluation script  
│   ├── reptile.py         # Reptile meta-learning implementation
│   └── [other scripts]
├── datasets/              # Processed data files
├── results/               # Training and evaluation results
└── environment.yml       # Conda environment specification
```
