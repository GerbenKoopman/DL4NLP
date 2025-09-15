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
