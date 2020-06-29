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
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Automatic Question Generation Training')
parser.add_argument('--train-set', default='../dataset/squad_train.csv', type=str, metavar='PATH',
                    help='path to train dataset')
parser.add_argument('--dev-set', default='../dataset/squad_dev.csv', type=str, metavar='PATH',
                    help='path to dev dataset')
parser.add_argument('--test-size', default=0.7, type=int, metavar='N',
                    help='ratio of test set from the whole dev_set')
parser.add_argument('--save', default='../dataset/', type=str, metavar='PATH',
                    help='save path to extracted features folder')
parser.add_argument('--word-vector', default='glove', type=str,
                    help='word vector to use(glove or numberbatch,;if numberbatch add location folder with --numberbatch-loc)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--numberbatch-loc', default='../dataset/', type=str, metavar='PATH',
                    help='folder of numberbatch word embeddings')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs for training')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Split dev dataset into test set and validation set
dev_set = pd.read_csv(args.dev_set)
validation_set, test_set = train_test_split(dev_set, test_size = args.test_size)

# Saving file names to variables
trainloc = args.train_set
valloc = args.save+'validation_set.csv'
testloc = args.save+'test_set.csv'

# Saving validation and test set to csv file
validation_set.to_csv(valloc, index=False)
test_set.to_csv(testloc, index=False)

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

# Create a set of iterators for each split
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x:len(x.context),
     device = device)

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

# If continuing training
if (args.resume):
	model.load_state_dict(torch.load(args.resume))

def train(model, iterator, criterion, optimizer, clip):
    # Put the model in training mode
    model.train()
    
    epoch_loss = 0
    
    for idx, batch in tqdm(enumerate(iterator), total=len(iterator)):
        
        input_sequence = batch.context
        answer_sequence = batch.bio
        output_sequence = batch.question
        lexical_sequence = batch.lex
        
        target_tokens = output_sequence[0]
        
        # zero out the gradient for the current batch
        optimizer.zero_grad()
        
        # Run the batch through the model
        output = model(input_sequence, answer_sequence, lexical_sequence, output_sequence, 0.5)
        
        # Throw it through the loss function
        output = output[1:].view(-1, output.shape[-1])
        target_tokens = target_tokens[1:].view(-1)
        
        loss = criterion(output, target_tokens)
        
        # Perform back-prop and calculate the gradient of the loss function
        loss.backward()
          
        # Clip the gradient if necessary.          
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update model parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    # Put model in evaluation mode
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for idx, batch in tqdm(enumerate(iterator), total=len(iterator)):

            input_sequence = batch.context
            answer_sequence = batch.bio
            output_sequence = batch.question
            lexical_sequence = batch.lex
            
            target_tokens = output_sequence[0]
            
            # Run the batch through the model
            output = model(input_sequence, answer_sequence, lexical_sequence, output_sequence, 0)
            
            # Throw it through the loss function
            output = output[1:].view(-1, output.shape[-1])
            target_tokens = target_tokens[1:].view(-1)

            loss = criterion(output, target_tokens)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

N_EPOCHS = args.epochs
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    train_loss = train(model, train_iterator, criterion, optimizer, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), args.save+'/model.pth')

    print('Epoch: ', epoch)
    print('Train loss: ', train_loss)
    print('Valid loss: ', valid_loss)

test_loss = evaluate(model, test_iterator, criterion)

print('Test Loss: {:.2f}'.format(test_loss))

for instance in list(tqdm._instances):
    tqdm._decr_instances(instance)