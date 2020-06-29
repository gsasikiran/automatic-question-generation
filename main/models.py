# code source : ''https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        # Here input_size refers to the number of vocabulary
        # hidden_size is the dimension of embedding
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded

        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size, device = device)

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p = 0.5):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim = 1)
        return output, hidden, attn_weights

    def  initHidden(self):
        return torch.zeros(1,1, self.hidden_size, device = device)

class EncoderBiGRU(nn.Module):

    def __init__(self, input_size:int, hidden_size:int, embedding_dim:int, embeddings: np.array = None,
                 n_layers: int = 1, dropout: float = 0.5):
        # Here input_size refers to the number of vocabulary
        # hidden_size is the dimension of embedding
        super(EncoderBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.gru_dropout = dropout

        self.embedding = nn.Embedding(self.input_size, self.embedding_dim)

        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
            self.embedding.requires_grad = False

        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size // 2,
                          num_layers=self.n_layers, bidirectional=True, dropout=self.gru_dropout)

    def forward(self, inputs, lengths, return_packed=False):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embeds, lengths=lengths, batch_first=True)

        outputs, hiddens = self.gru(packed)
        if not return_packed:
            return pad_packed_sequence(outputs, True)[0], hiddens
        return outputs, hiddens

class DecoderLSTM(nn.Module):
    """
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, n_layers: int = 1,
                 encoder_hidden_dim: int = None, embeddings: np.array = None,
                 dropout:float = 0.2):
        super(DecoderLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dropout=dropout
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

        assert embeddings is not None
        self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embeds.requires_grad=False

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,dropout=self.lstm_dropout)

        # h_t^T W h_s
        self.linear_out = nn.Linear(hidden_dim, vocab_size)
        self.attn = GlobalAttention(encoder_hidden_dim, hidden_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, context, context_lengths, eval_mode=False):
        """
        inputs: (tgt_len, batch_size, d)
        hidden: last hidden state from encoder
        context: (src_len, batch_size, hidden_size), outputs of encoder
        """
        embedded = self.word_embeds(inputs)
        embedded = embedded.transpose(0, 1)
        if not eval_mode:
            if self.n_layers==2:
                decode_hidden_init = torch.stack((torch.cat([hidden[0][0], hidden[0][1]],1),torch.cat([hidden[0][2], hidden[0][3]], 1)),0)
                decode_cell_init = torch.stack((torch.cat([hidden[1][0], hidden[1][1]],1),torch.cat([hidden[1][2], hidden[1][3]], 1)),0)
            else :
                decode_hidden_init = torch.cat([hidden[0][2], hidden[0][3]], 1).unsqueeze(0)
                decode_cell_init = torch.cat([hidden[1][2], hidden[1][3]], 1).unsqueeze(0)

        else:
            decode_hidden_init = hidden[0]
            decode_cell_init = hidden[1]


        # embedded = self.dropout(embedded)
        decoder_unpacked, decoder_hidden = self.lstm(embedded, (decode_hidden_init,decode_cell_init))
        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            decoder_unpacked.transpose(0, 1).contiguous(),  # (len, batch, d) -> (batch, len, d)
            context,  # (len, batch, d) -> (batch, len, d)
            context_lengths=context_lengths
        )

        outputs = self.linear_out(attn_outputs)
        return outputs, decoder_hidden