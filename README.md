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
```