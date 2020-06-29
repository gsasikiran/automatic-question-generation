# AUTOMATIC QUESTION GENERATION

This directory contains a pytorch implementation of all SQuAD experiments based on the following paper  
[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/pdf/1704.01792.pdf) (EMNLP 2017).

## Dependencies
torch v1.5, spacy 2.2.4, torchtext 0.3.1

## Download SQuAD dataset

The `directory` argument specifies which directory to store dataset.
```shell
python main.py --directory ../dataset
```

## Squad Parse

Parsing SQuAD v.2.0 dataset and preprocess data by obtaining the lexical features (POS-Tagging, NER, and case) and IOB tagging.
```shell
python squad_parse.py --train_filepath [PATH TO TRAIN JSON] --dev_filepath [PATH TO DEV JSON] --save [DIRECTORY TO STORE RESULT]
```

## Train

Train the model.
```shell
python train.py --train-set [PATH TO THE PREPROCESSED TRAIN CSV] --dev-set [PATH TO THE PREPROCESSED DEV CSV] --epochs 100  --save [DIRECTORY TO STORE RESULT] --batch-size 128 --save [DIRECTORY TO STORE RESULT]
```
More flags in this file:
- `--resume` flag to continue from previous training
- `--word-vector` word embeddings to choose; currently support [GloVe](https://nlp.stanford.edu/projects/glove/) and [ConceptNet-Numberbatch](https://github.com/commonsense/conceptnet-numberbatch)
- `--numberbatch-loc` location of downloaded numberbatch word embeddings
- `--test-size` size of test ratio split from dev set
- `--batch-size` size of batch for training


## Evaluate
Generating N number of predictions and evaluate the predictions based on BLEU.
```shell
python evaluate.py --train-set [PATH TO THE PREPROCESSED CSV] --display 100 --load [DIRECTORY TO TRAINED MODEL] --data-folder [DIRECTORY TO VAL AND TEST DATA]
```
More flags in this file:
-`--word-vector` word embeddings to choose; currently support [GloVe](https://nlp.stanford.edu/projects/glove/) and [ConceptNet-Numberbatch](https://github.com/commonsense/conceptnet-numberbatch)
- `--numberbatch-loc` location of downloaded numberbatch word embeddings
- `--batch-size` size of batch for training
