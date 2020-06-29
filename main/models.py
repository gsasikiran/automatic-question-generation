# References: https://medium.com/@adam.wearne/seq2seq-with-pytorch-46dc00ff5164

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

class Encoder(nn.Module):
  
    def __init__(self, hidden_size, embedding_size,
                 embedding, answer_embedding, lexical_embedding, n_layers, dropout):
      
        super(Encoder, self).__init__()
        
        # Initialize network parameters
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Embedding layer to be shared with Decoder
        self.embedding = embedding
        self.answer_embedding = answer_embedding
        self.lexical_embedding = lexical_embedding
        
        # Bidirectional GRU
        self.gru = nn.GRU(embedding_size, hidden_size,
                          num_layers=n_layers,
                          dropout=dropout,
                          bidirectional=True)
        
    def forward(self, input_sequence, input_lengths, answer_sequence, lexical_sequence):
        
        # Convert input_sequence to word embeddings
        word_embeddings = self.embedding(input_sequence)
        answer_embeddings = self.answer_embedding(answer_sequence)
        lexical_embeddings = self.lexical_embedding(lexical_sequence)

        # Concatenate word embeddings from all features
        final_embeddings = torch.cat((word_embeddings,answer_embeddings,lexical_embeddings), 0)
        
        # Pack the sequence of embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(final_embeddings, input_lengths)
        
        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, hidden = self.gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        # The ouput of a GRU has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the 
        # forward and reversed sequence by simply adding them together.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        
    def dot_score(self, hidden_state, encoder_states):
        # Attention model use the dot product formula as global attention
        return torch.sum(hidden_state * encoder_states, dim=2)
    
    def forward(self, hidden, encoder_outputs, mask):
        attn_scores = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_scores = attn_scores.t()
        
        # Apply mask so network does not attend <pad> tokens        
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        
        # Return softmax over attention scores      
        return F.softmax(attn_scores, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, embedding, embedding_size,
                 hidden_size, output_size, n_layers, dropout):
        
        super(Decoder, self).__init__()
        
        # Initialize network params
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding
                
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, 
                          dropout=dropout)
        
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)
        
    def forward(self, current_token, hidden_state, encoder_outputs, mask):
      
        # convert current_token to word_embedding
        embedded = self.embedding(current_token)
        
        # Pass through GRU
        rnn_output, hidden_state = self.gru(embedded, hidden_state)
        
        # Calculate attention weights
        attention_weights = self.attn(rnn_output, encoder_outputs, mask)
        
        # Calculate context vector
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # Concatenate  context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Pass concat_output to final output layer
        output = self.out(concat_output)
        
        # Return output and final hidden state
        return output, hidden_state

class Seq2seq(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, 
                 device, pad_idx, eos_idx, sos_idx, teacher_forcing_ratio=0.5):
        super(Seq2seq, self).__init__()
        
        # Initialize embedding layer shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.answer_embedding = nn.Embedding(6, embedding_size, padding_idx=1)
        # Size could sometime change, depend on the device that the model is trained on
        self.lexical_embedding = nn.Embedding(452, embedding_size, padding_idx=1)
        
        # Encoder network
        self.encoder = Encoder(hidden_size, 
                               embedding_size, 
                               self.embedding,
                               self.answer_embedding,
                               self.lexical_embedding,
                               n_layers=2,
                               dropout=0.5)
        
        # Decoder network        
        self.decoder = Decoder(self.embedding,
                               embedding_size,
                               hidden_size,
                               vocab_size,
                               n_layers=2,
                               dropout=0.5)
        
        
        # Indices of special tokens and hardware device 
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx
        self.device = device
        
    def create_mask(self, input_sequence):

        return (input_sequence != self.pad_idx).permute(1, 0)
        
    def forward(self, input_sequence, answer_sequence, lexical_sequence, output_sequence, teacher_forcing_ratio):
      
        # Unpack input_sequence tuple
        input_tokens = input_sequence[0]
        input_lengths = input_sequence[1]
      
        # Unpack output_tokens, or create an empty tensor for text generation
        if output_sequence is None:
            inference = True
            output_tokens = torch.zeros((100, input_tokens.shape[1])).long().fill_(self.sos_idx).to(self.device)
        else:
            inference = False
            output_tokens = output_sequence[0]
        
        vocab_size = self.decoder.output_size
        
        batch_size = len(input_lengths)
        max_seq_len = len(output_tokens)
        
        # Tensor initialization to store Decoder output
        outputs = torch.zeros(max_seq_len, batch_size, vocab_size).to(self.device)
        
        # Pass through the first half of the network
        encoder_outputs, hidden = self.encoder(input_tokens, input_lengths, answer_sequence, lexical_sequence)
        
        # Ensure dim of hidden_state can be fed into Decoder
        hidden =  hidden[:self.decoder.n_layers]
        
        # First input to the decoder is the <sos> tokens
        output = output_tokens[0,:]
        
        # Create mask
        mask = self.create_mask(input_tokens)
        
        # Step through the length of the output sequence one token at a time
        # Teacher forcing is used to assist training
        for t in range(1, max_seq_len):
            output = output.unsqueeze(0)
            
            output, hidden = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (output_tokens[t] if teacher_force else top1)
            
            # If we're in inference mode, keep generating until we produce an
            # <eos> token
            if inference and output.item() == self.eos_idx:
                return outputs[:t]
        
        return outputs