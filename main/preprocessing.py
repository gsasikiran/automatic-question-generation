import json
import pandas as pd
import numpy as np
import spacy

def preprocess(file_name):
    '''
    preprocess SQuAD dataset and return list of examples,
    which contain list of one example containing tokenized context and questions,
    original answer, question id, and whether it is impossible (one question contain one answer)

    Arguments:
        file_name -- location of dataset file
    Returns:
        examples  -- list of example data from dataset
    '''
	nlp = spacy.load('en_core_web_sm')
	examples = []
	idq = 0

	with open(file_name) as f:
	    json_data = json.load(f)['data']
	    for i in range(len(json_data)):
	        article = json_data[i]
	        for par in article['paragraphs']:
	            context = par['context'].replace("``",'" ')
	            context_tokens = [token.text for token in nlp(context)]
	            for qa in par['qas']:
	                idq += 1
	                question = qa['question'].replace("``",'" ')
	                question_tokens = [token.text for token in nlp(question)]
	                answer_text = []
	                for ans in qa['answers']:
	                    answer = ans['text']
	                    answer_text.append(answer)
	                example = {"id": idq,
	                           "context_tokens": context_tokens,
	                           "question_tokens": question_tokens,
	                           "original_answer": answer_text,
	                           "is_impossible": qa['is_impossible']}
	                examples.apppend(example)
	return examples

preprocess('../dataset/train-v2.0.json')
