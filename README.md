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

We intend to use Reptile meta-learning, with LoRA if this proves necessary for the finetuning of a model like Gemma 3 1B or Gemma 3 270m.

## Example Start

```bash
# 1. Activate environment
conda env create -f environment.yaml
conda activate dl4nlp

# 2. Data preparation (already done)
python3 clean_data.py --data_dir datasets --split all

# 3. Run full baseline evaluation
python3 baseline_eval.py --model 1b --data_dir datasets --output results/baseline_1b.json --max_examples 100

# 4. Train Reptile meta-learning
python3 train_reptile.py --model 1b --data_dir datasets --output_dir results/reptile_1b --meta_steps 50 --inner_steps 5

# 5. Evaluate meta-learned model
python3 evaluate_reptile.py --model 1b --data_dir datasets --output_dir results/eval_1b --support_size 5

# 6. Compare with baseline
python3 evaluate_reptile.py --model 1b --data_dir datasets --output_dir results/comparison_1b --baseline_file results/baseline_1b.json

# 7. Generate plots
python3 analyze_results.py --results_dir results --plots_dir results/plots
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

### TODO

- [ ] Implement wandb logging for training monitoring
- [ ] Add QLoRA support for efficient fine-tuning ([docs](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora))
- [ ] Rewrite Reptile to update QLoRA weights and parameters correctly
- [ ] Expand evaluations
