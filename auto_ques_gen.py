# Code adapted from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
import pickle

import torch

from main.models import DecoderRNN, AttnDecoderRNN
from main.models import EncoderRNN
from main.training import TrainIters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH_PAIRS = 'dataset/updated/input_output_pairs.pickle'
PATH_PAIRS_INT = 'dataset/updated/input_output_pairs_int.pickle'
PATH_INT2VOCAB = "dataset/updated/int2vocab.pickle"
PATH_VOCAB2INT = "dataset/updated/vocab2int.pickle"

PATH_PRE_INPUT = "dataset/updated/preprocessed_inputs.pickle"
PATH_PRE_ANS = "dataset/updated/preprocessed_inputs_ans.pickle"
PATH_PRE_TAR = "dataset/updated/preprocessed_targets.pickle"

PATH_VOCAB = "dataset/updated/vocab_list.pickle"
PATH_WORD_EMBED = "dataset/updated/preprocessed_targets.pickle"

pairs = pickle.load(open(PATH_PAIRS, "rb"))
pairs_int = pickle.load(open(PATH_PAIRS_INT, "rb"))

int2vocab = pickle.load(open(PATH_INT2VOCAB, "rb"))
vocab2int = pickle.load(open(PATH_VOCAB2INT, "rb"))

vocab_list = pickle.load(open(PATH_VOCAB, "rb"))
word_embeddings = pickle.load(open(PATH_WORD_EMBED, "rb"))  #

preprocess_input = pickle.load(open(PATH_PRE_INPUT, "rb"))
preprocess_target = pickle.load(open(PATH_PRE_TAR, "rb"))
preprocess_answer = pickle.load(open(PATH_PRE_ANS, "rb"))

vocab_size = len(vocab_list)
hidden_size = 256

max_length = 733 # Calculated value from the input

print('Assigning encoder and decoder ...')
encoder = EncoderRNN(vocab_size, hidden_size).to(device)
seq2seq_decoder = DecoderRNN(hidden_size, vocab_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, vocab_size, max_length ,dropout_p=0.1).to(device)

print('Started training ...')
train_iters = TrainIters()
train_iters.train(encoder, attn_decoder, pairs_int, 75000, max_length, is_attention=True)

print('Saving models ...')
torch.save(encoder.state_dict(), 'weights/encoder.dict()')
torch.save(encoder.state_dict(), 'weights/attndecoder.dict()')

print('Done ...')
