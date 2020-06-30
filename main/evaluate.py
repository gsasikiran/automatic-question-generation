# Code adapted from : https://github.com/bentrevett/pytorch-seq2seq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext.vocab import Vectors

from tqdm import tqdm
import argparse
from models import Seq2seq
import random
import pandas as pd
import numpy as np

import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
## Wordnet dependencies from meteor score
#nltk.download('wordnet')

parser = argparse.ArgumentParser(description='Automatic Question Generation Evaluator')
parser.add_argument('--train-set', default='../dataset/squad_train.csv', type=str, metavar='PATH',
                    help='path to train dataset')
parser.add_argument('--word-vector', default='glove', type=str,
                    help='word vector to use(glove or numberbatch,;if numberbatch add location folder with --numberbatch-loc)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--numberbatch-loc', default='../dataset/', type=str, metavar='PATH',
                    help='folder of numberbatch word embeddings')
parser.add_argument('--load', default='./model.pth', type=str, metavar='PATH',
                    help='path to model (default: ./model.pth)')
parser.add_argument('--display', default=100, type=int, metavar='N',
                    help='number of display for prediction')
parser.add_argument('--data-folder', default='../dataset/', type=str, metavar='PATH',
                    help='path to val and test set')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
trainloc = args.train_set
valloc = args.data_folder+'validation_set.csv'
testloc = args.data_folder+'test_set.csv'

# Create Field object
tokenize = lambda x: x.split()
TEXT = data.Field(tokenize=tokenize, lower=False, include_lengths = True, init_token = '<SOS>', eos_token = '<EOS>')
LEX = data.Field(tokenize=tokenize, lower=False, init_token = '<SOS>', eos_token = '<SOS>')
BIO = data.Field(tokenize=tokenize, lower=False, init_token = '<SOS>', eos_token = '<SOS>')

# Specify Fields in the dataset
fields = [('context', TEXT), ('question', TEXT), ('bio', BIO), ('lex', LEX)]

# Build the dataset
train_data, valid_data, test_data = data.TabularDataset.splits(path = '',train=trainloc, validation=valloc,
                                 test=testloc, fields = fields, format='csv', skip_header=True)

# Build vocabulary
MAX_VOCAB_SIZE = 50000
MIN_COUNT = 5
BATCH_SIZE = args.batch_size

if args.word_vector == 'glove':
  TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE,
                 min_freq=MIN_COUNT, vectors='glove.6B.300d',
                 unk_init=torch.Tensor.normal_)
else:
  cache_ = args.numberbatch_loc 
  vectors = Vectors(name='numberbatch-en-19.08.txt', cache=cache_)
  TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE,
                 min_freq=MIN_COUNT, vectors=vectors,
                 unk_init=torch.Tensor.normal_)

BIO.build_vocab(train_data)
LEX.build_vocab(train_data)

# Building model
pad_idx = TEXT.vocab.stoi['<pad>']
eos_idx = TEXT.vocab.stoi['<EOS>']
sos_idx = TEXT.vocab.stoi['<SOS>']

# Size of embedding_dim should match the dim of pre-trained word embeddings
embedding_dim = 300
hidden_dim = 512
vocab_size = len(TEXT.vocab)

# Initializing weights
model = Seq2seq(embedding_dim, hidden_dim, vocab_size, device, pad_idx, eos_idx, sos_idx).to(device)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Initializing weights for special tokens
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

model.embedding.weight.requires_grad = False

optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], 
                       lr=1.0e-3)
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

# Load model
model.load_state_dict(torch.load(args.load))

def predict_question(model, paragraph, answer_pos, lex_features):
    model.eval()
    
    tokenized = ['<SOS>'] + paragraph + ['<EOS>']
    numericalized = [TEXT.vocab.stoi[t] for t in tokenized] 

    tokenized_answer = ['<SOS>'] + answer_pos + ['<EOS>']
    numericalized_answer = [BIO.vocab.stoi[t] for t in tokenized_answer] 

    tokenized_lex = ['<SOS>'] + lex_features + ['<EOS>']
    numericalized_lex = [LEX.vocab.stoi[t] for t in tokenized_lex]
    
    paragraph_length = torch.LongTensor([len(numericalized)]).to(model.device) 
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(model.device) 
 
    answer_tensor = torch.LongTensor(numericalized_answer).unsqueeze(1).to(model.device) 
    lex_tensor = torch.LongTensor(numericalized_lex).unsqueeze(1).to(model.device)
    
    question_tensor_logits = model((tensor, paragraph_length), answer_tensor, lex_tensor, None, 0) 
    
    question_tensor = torch.argmax(question_tensor_logits.squeeze(1), 1)
    question = [TEXT.vocab.itos[t] for t in question_tensor]
 
    # Start at the first index.  We don't need to return the <SOS> token
    question = question[1:]

    return question, question_tensor_logits

# Display prediction
num = args.display
example_idx = random.sample(range(1,300),num)

for i in example_idx:
  src = vars(train_data.examples[i])['context']
  trg = vars(train_data.examples[i])['question']
  ans = vars(train_data.examples[i])['bio']
  lex = vars(train_data.examples[i])['lex']

  print('context: ', ' '.join(src))
  print('question: ', ' '.join(trg))

  question, logits = predict_question(model, src, ans, lex)
  print('predicted: ', " ".join(question))
  print()

for j in example_idx:
  src = vars(test_data.examples[j])['context']
  trg = vars(test_data.examples[j])['question']
  ans = vars(test_data.examples[j])['bio']
  lex = vars(test_data.examples[j])['lex']

  print('context: ', ' '.join(src))
  print('question: ', ' '.join(trg))

  question, logits = predict_question(model, src, ans, lex)
  print('predicted: ', " ".join(question))
  print()

def calculate_bleu_and_meteor(data, model):
    
    trgs = []
    pred_trgs = []
    meteor_score_ = []
    
    for datum in data:
        
        src = vars(datum)['context']
        trg = vars(datum)['question']
        ans = vars(datum)['bio']
        lex = vars(datum)['lex']
        
        pred_trg, _ = predict_question(model, src, ans, lex)
        
        #cut off <EOS> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        # print(pred_trg)
        trgs.append(trg)
        # print(trg)
        meteor_score_.append(single_meteor_score(' '.join(pred_trg),' '.join(trg)))
        
    bleu_score = corpus_bleu(pred_trgs, trgs)
    meteor_score_ = np.mean(meteor_score_)
    
    return bleu_score,meteor_score_

bleu_score, meteor_score_ = calculate_bleu_and_meteor(test_data, model)

print('BLEU score = {:.2f}'.format(bleu_score*100))
print('METEOR score = {:.2f}'.format(meteor_score_*100))