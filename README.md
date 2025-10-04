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
- **Models**: Gemma 3 variants - 270M, 1B (primary), and 4B (`google/gemma-3-*-it`)
- **Meta-learning**: Reptile algorithm for few-shot adaptation
- **Fine-tuning**: QLoRA (Quantized Low-Rank Adaptation) with 4-bit quantization (CUDA) or FP16 (MPS/CPU)
- **Monitoring**: Weights & Biases (wandb) integration for training metrics (BLEU, chrF, meta-average)
- **Evaluation**: Comprehensive evaluation suite (zero-shot, few-shot, transfer)
- **Ablation**: Systematic ablation study infrastructure to quantify component contributions

> **Note**: 1B is recommended for development; 4B for final runs with CUDA GPUs (requires ~20-24GB VRAM with 4-bit quantization).

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

# 5. Evaluate meta-learned model (loads adapter from results/adapters/...)
python evaluate_reptile.py --model 1b --support_size 5 --adapter_mode all

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
  --meta_lr 0.1 \
  --bleu_weight 0.6 \
  --seed 42 \
  --output_dir results/reptile_1b

# Train 4B model (requires CUDA GPU with ~24GB VRAM)
python train_reptile.py \
  --model 4b \
  --meta_steps 50 \
  --inner_steps 3 \
  --support_size 5 \
  --adapter_mode all

# Evaluate trained adapter (default adapter path under output_dir/adapters)
python evaluate_reptile.py \
  --model 4b \
  --support_size 5 \
  --adapter_mode all \
  --adapter_dir results/adapters/gemma-3-4b-it_all
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
| `train_reptile.py` | Train Reptile meta-learning | `--model`, `--meta_steps`, `--adapter_mode`, `--meta_lr`, `--seed` |
| `evaluate_reptile.py` | Evaluate trained models | `--model`, `--baseline_file`, `--support_size`, `--seed` |
| `baseline_eval.py` | Run baseline evaluations | `--model`, `--max_examples` |
| `clean_data.py` | Process TED Talks data | `--split` (train/dev/test/all) |
| `analyze_results.py` | Generate plots and analysis | `--train_tasks`, `--eval_types`, `--ablation` |
| `run_ablation_study.py` | Run ablation experiments | `--model`, `--ablation_type`, `--skip_training`, `--aggregate_only` |

## Key Features

✅ **Implemented:**
- QLoRA fine-tuning with 4-bit quantization for memory efficiency (CUDA) or FP16 (MPS/CPU)
- Reptile meta-learning algorithm adapted for QLoRA weights
- Weights & Biases integration with BLEU/chrF metrics logging  
- Comprehensive evaluation suite (zero-shot, few-shot, transfer)
- Flexible adapter modes (all languages, az_en, be_en)
- Automated results analysis and visualization
- **Ablation study infrastructure** with systematic grid search
- **Multi-scale support** (270M, 1B, 4B models)
- **Reproducible experiments** with seed control and configurable metrics

🔄 **In Development:**
- Additional meta-learning baselines for comparison (MAML, ProtoNet)

## Ablation Study

### Running Ablation Experiments

The ablation study systematically tests which components of the Reptile+QLoRA pipeline contribute to performance:

**Minimal ablation** (recommended first pass, 16 configs):
```bash
cd src
python run_ablation_study.py --model 1b --ablation_type minimal
```

**Extended ablation** (more comprehensive):
```bash
python run_ablation_study.py --model 1b --ablation_type extended
```

**Metric sensitivity only**:
```bash
python run_ablation_study.py --model 1b --ablation_type metric_only
```

**Aggregate and visualize existing results**:
```bash
python run_ablation_study.py --aggregate_only
python analyze_results.py --ablation --ablation_metric transfer_5
```

### Ablation Factors Tested

| Factor | Values | Hypothesis |
|--------|--------|------------|
| `meta_lr` | 0.0, 0.05, 0.1 | Does Reptile meta-update improve over pure adaptation? |
| `inner_steps` | 0, 1, 3, 5 | How much inner-loop training is needed? |
| `support_size` | 1, 5 | 1-shot vs 5-shot efficiency |
| `adapter_mode` | az_en, be_en, all | Does multi-family training help transfer? |
| `bleu_weight` | 0.5, 0.6 | Metric weighting sensitivity |
| `episodes_per_task` | 1, 3 | Variance vs compute tradeoff |

### Interpreting Results

- **meta_lr=0.0**: Control for "no meta-learning" (pure few-shot adaptation)
- **inner_steps=0**: Control for "no adaptation" (zero-shot baseline)
- **adapter_mode**: Tests whether training on both language families (az_en + be_en) improves transfer to both target families (Turkish, Ukrainian)

Results are saved to:
- `results/ablation/ablation_summary_*.json` - Full experiment log
- `results/ablation/aggregated_results_*.csv` - All eval scores in table format
- `results/plots/ablation_comparison_*.png` - Visual comparisons
- `results/plots/ablation_factor_effects.png` - Individual factor effects

## Scaling to Larger Models

### Gemma 3 4B Model

The 4B model offers ~4× capacity vs 1B with QLoRA fine-tuning feasibility:

**Requirements:**
- CUDA GPU with ≥24GB VRAM (e.g., RTX 4090, A5000, A6000)
- 4-bit quantization enabled (automatic on CUDA)
- Typical memory: ~8-12GB base model + ~8-12GB optimizer/activations with LoRA

**Training 4B:**
```bash
cd src
python train_reptile.py \
  --model 4b \
  --meta_steps 50 \
  --inner_steps 3 \
  --support_size 5 \
  --adapter_mode all
```

**Evaluation:**
```bash
python evaluate_reptile.py --model 4b --support_size 5
```

**Baseline for comparison:**
```bash
python baseline_eval.py --model 4b --max_examples 50
```

**Memory optimization tips** (if OOM):
- Reduce `max_length` in `ReptileConfig` (default 128)
- Set `per_device_train_batch_size=1` (default already 1)
- Reduce LoRA `r` from 16 to 8 in `QLoRAConfig`
- Enable gradient checkpointing in `TrainingArguments`

### Performance vs Cost Tradeoffs

| Model | Params | VRAM (4-bit) | Training Time* | Expected BLEU Δ** |
|-------|--------|--------------|----------------|-------------------|
| 270M  | 270M   | ~4-6 GB      | 1× (baseline)  | Baseline          |
| 1B    | 1B     | ~8-12 GB     | ~2-3×          | +5-10 pts         |
| 4B    | 4B     | ~20-24 GB    | ~6-8×          | +3-7 pts over 1B  |

\* Relative to 270M on same hardware; \*\* Speculative estimates based on typical scaling laws

## Project Structure

```
DL4NLP/
├── src/                       # Source code
│   ├── train_reptile.py       # Main training script
│   ├── evaluate_reptile.py    # Evaluation script  
│   ├── reptile.py             # Reptile meta-learning implementation
│   ├── run_ablation_study.py  # Ablation experiment runner
│   ├── analyze_results.py     # Results analysis and plotting
│   └── [other scripts]
├── datasets/                  # Processed data files
├── results/                   # Training and evaluation results
│   ├── ablation/              # Ablation study results
│   ├── adapters/              # Trained LoRA adapters
│   └── plots/                 # Generated visualizations
└── environment.yml            # Conda environment specification
```
