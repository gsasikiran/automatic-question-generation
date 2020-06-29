# AUTOMATIC QUESTION GENERATION

This directory contains a pytorch implementation of all SQuAD experiments based on the following paper  
[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/pdf/1704.01792.pdf) (EMNLP 2017).

## Dependencies
torch v1.5, spacy 2.2.4

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
python train.py --preprocessed [PATH TO THE PREPROCESSED CSV] --epochs 100 --word_vector glove --save [DIRECTORY TO STORE RESULT] --batch-size 128 --save [DIRECTORY TO STORE RESULT]
```

We can add `--resume` flag to continue from previous training.