#Code adapted from https://github.com/8horn/seq2seq/blob/master/parse_squad.py
import json
import pandas as pd

def squad_parse(file_name):
    '''
    preprocess SQuAD dataset and return list of examples,
    which contain list of one example containing context and questions, and answer

    Arguments:
        file_name -- location of dataset file
    Returns:
        examples  -- list of example data from dataset
    '''
	examples = []
	idq = 0

	with open(file_name) as f:
	    json_data = json.load(f)['data']
	    for i in range(len(json_data)):
	        article = json_data[i]
	        for par in article['paragraphs']:
	            context = par['context'].replace("``",'" ')
	            for qa in par['qas']:
	                question = qa['question'].replace("``",'" ')
	                answer_text = []
	                for ans in qa['answers']:
	                    answer = ans['text']
	                    answer_text.append(answer)
	                example = {"topic": article['title']
	                		   "context": context,
	                           "question": question,
	                           "answers": answer_text}
	                examples.apppend(example)
	return examples

# Parsing and saving for train data
parsed_train = squad_parse('../dataset/train-v2.0.json')
json.dump(parsed_train, open('../dataset/parsed_train.json', 'w'))

# Parsing and saving dev data
parsed_dev = squad_parse('../dataset/dev-v2.0.json')
json.dump(parsed_dev, open('../dataset/parsed_dev.json', 'w'))

# Extract paragraph,answers and questions pairs from SQuAD dataset
train_set =  json.load(open('../dataset/parsed_train.json','r'))
dev_set =  json.load(open('../dataset/parsed_dev.json','r'))

train_pairs = [[section['context'],section['answers'],section['question']] for section in train_set]
dev_pairs = [[section['context'],section['answers'],section['question']] for section in dev_set]

# Saving pairs to csv format
df_train = pd.DataFrame(train_pairs)
df_train.columns = ['Paragraph','Answers','Question']
df_train.to_csv('../dataset/train_pairs.csv', encoding='utf-8')

df_dev = pd.DataFrame(dev_pairs)
df_dev.columns = ['Paragraph','Answers','Question']
df_dev.to_csv('../dataset/dev_pairs.csv', encoding='utf-8')

