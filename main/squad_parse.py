import pandas as pd
import string
import json
import random
import spacy
import argparse

pd.options.mode.chained_assignment = None  # remove unnecessary warnings

parser = argparse.ArgumentParser(description='SQuAD Dataset Parser and Feature Extractor')
parser.add_argument('--train_filepath', default='../dataset/train-v2.0.json', type=str, metavar='PATH',
                    help='path to train dataset')
parser.add_argument('--dev_filepath', default='../dataset/dev-v2.0.json', type=str, metavar='PATH',
                    help='path to dev dataset')
parser.add_argument('--save', default='../dataset/', type=str, metavar='PATH',
                    help='save path to extracted features folder')

args = parser.parse_args()

def squad_parse(file_path):
    '''
    Parse SQuAD dataset and return list of examples,
    which contain list of context, questions, answer, title,
    indicator of available answer, and start index of answer

    Arguments:
        file_path -- location of dataset file
    Returns:
        examples  -- pandas dataframe parsed data from dataset
    '''

    # Load json file of dataset 
    f = open(file_path)
    doc_ = json.load(f)
    doc = pd.json_normalize(data=doc_['data'], record_path=['paragraphs'], meta=['title'])
    f.close()

    examples = pd.DataFrame()
    for idx, data in doc[['qas']].itertuples():
    	# Create new dictionary from dataset
        example = pd.DataFrame.from_dict(data)
        example['title'] = doc['title'][idx]
        example['context'] = doc['context'][idx]
        example['answer'] = ''
        example['answer_start'] = ''

        for i, answer in example[['answers']].itertuples():
        	# If answer is not available, take one of plausible answer
            if answer == []:
                answer = example['plausible_answers'][i]
            try:
                dic = random.sample(answer, 1)[0]
            except ValueError:
                pass
            example['answer'][i] = dic.get('text')

            # Cleaning answers from punctuations
            example['answer'][i] = example['answer'][i].strip(string.punctuation)
            example['answer'][i] = example['answer'][i].strip()

            example['answer_start'][i] = dic.get('answer_start')
        examples = examples.append(example)

    # Removing answers, plausible_answers, id, and index column
    examples = examples.drop('answers', axis=1)
    examples = examples.drop('plausible_answers', axis=1)   
    examples = examples.drop('id', axis=1)
    examples = examples.reset_index()
    examples = examples.drop('index', axis=1)
    
    return examples

nlp = spacy.load("en_core_web_sm")

def extract_features(text, answer, answer_start, nlp):
    '''
    Extract answers to obtain POS, NER, case, BIO features based on text

    Arguments:
        text	-- context or paragraph
        answer 	-- answer in paragraph's question
        answer_start -- starting index of answer
        nlp 	-- spacy tool for nlp
    Returns:
        pos 	-- sequence of string of answer tokens part-of-speech tagging
		ner 	-- sequence of string of answer tokens named entity recognition
		case	-- sequence of string of answer tokens case
		bio 	-- sequence of string of answer tokens inside-outside-beggining tagging
		tokenized 	-- joined tokenized context (paragraph) with lower typecasting
    '''
    
    # Extract answer location index (left, right and answers itself) in text
    left = text[0:answer_start]
    ans = text[answer_start:answer_start+len(answer)+1]
    right = text[answer_start+len(answer)+1:len(text)+1]    
    
    # Initialize return values list
    pos = []
    ner = []
    case = []
    bio = []
    tokenized = []
    
    left_side = nlp(left)
    answer_range = nlp(ans)
    right_side = nlp(right)
    
    for token in left_side:
        if token.text != '' and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos.append(token.pos_)

            if token.ent_type_ == '':
                ner.append('O')
            else:
                ner.append(token.ent_type_)

            if token.text[0].isupper():
                case.append('UP')
            else:
                case.append('LOW')

            bio.append('O')
    
    for token in answer_range:
        if token.text != '' and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos.append(token.pos_)

            if token.ent_type_ == '':
                ner.append('O')
            else:
                ner.append(token.ent_type_)

            if token.text[0].isupper():
                case.append('UP')
            else:
                case.append('LOW')

            if token.i == 0:
                bio.append('B')
            else:
                bio.append('I')
    
    for token in right_side:
        if token.text != '' and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos.append(token.pos_)

            if token.ent_type_ == '':
                ner.append('O')
            else:
                ner.append(token.ent_type_)

            if token.text[0].isupper():
                case.append('UP')
            else:
                case.append('LOW')

            bio.append('O')
                
    return (' '.join(pos)), (' '.join(ner)), (' '.join(case)), (' '.join(bio)), (' '.join(tokenized))

def build_lexical_features(data):
    '''
    Creating pandas dataframe of features from parsed data

    Arguments:
        data -- data to be extracted; data must have context, answer, answer_start and question column
    Returns:
        data -- pandas dataframe of questions, context and features: IOB tag and lexical features(POS tag, NER, and case). 
    '''
    data['BIO'] = ''
    data['LEX'] = ''
    for idx, text, answer, answer_start, question in data[['context', 'answer', 'answer_start','question']].itertuples():
        pos, ner, case, data['BIO'][idx], data['context'][idx] = extract_features(text, str(answer), int(answer_start), nlp)
        lex = [i + '_' + j + '_' + k for i, j, k in zip(pos.split(), ner.split(), case.split())]
        data['LEX'][idx] = ' '.join(lex)
        data['question'][idx] = ' '.join([token.text.lower() for token in nlp(question)])

    # Building data on selected columns
    data = data[['context', 'question', 'BIO', 'LEX']]

    return data

if __name__ == '__main__':
	print('Processing training...')
	squad_train = squad_parse(args.train_filepath)
	squad_train = build_lexical_features(squad_train)
	trainloc = args.save + 'squad_train.csv'
	squad_train.to_csv(trainloc, index=False)
	print('Training done!')

	print('Processing dev..')
	squad_dev = squad_parse(args.dev_filepath)
	squad_dev = build_lexical_features(squad_dev)
	devloc = args.save + 'squad_dev.csv'
	squad_dev.to_csv(devloc, index=False)
	print('Dev done!')