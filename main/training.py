# code source : ''https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html'
import math
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

plt.switch_backend('agg')
import matplotlib.ticker as ticker

from models import EncoderRNN
from models import DecoderRNN, AttnDecoderRNN
from preprocessing import tensors_from_pairs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainArch:

    def __init__(self, encoder, encoder_optimizer,
                 decoder_optimizer, criterion, max_length, teacher_forcing_ratio=0.5):

        self.encoder = encoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

        self.criterion = criterion

        self.max_length = max_length
        self.SOS_token = 0
        self.EOS_token = 1
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def seq2seq(self, input_tensor, target_tensor, decoder):
        """
         Create loss from Seq2Seq architecture
        :param input_tensor: tensor
            Tensor of the input word
        :param target_tensor: tensor
            Tensor of the target word
        :param decoder: Seq2Seq decoder
        :return: tensor
            Returns the tensor of loss
        """

        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def attention(self, input_tensor, target_tensor, decoder):
        """
        Create loss from Attention architecture
        :param input_tensor: tensor
            Tensor of the input word
        :param target_tensor: tensor
            Tensor of the target word
        :param decoder: Attention decoder
        :return: tensor
            Returns the tensor of loss
        """

        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length
        pass


class TrainIters:

    def __init__(self):
        pass

    def __as_minutes(self, s):
        """
         Converts seconds into minutes
        :param s: float
            Time in seconds
        :return: float
            Time in minutes and seconds
        """
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __time_since(self, since, percent):
        """
            Calculates the time
        :param since: float
            Start time in seconds
        :param percent: int
            Percentage of time completed
        :return: float
            Time completed and remaining in minutes
        """
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (-%s)' % (self.__as_minutes(s), self.__as_minutes(rs))

    def __show_plot(self, points):
        """
            Line plot of the loss points
        :param points: list
            Loss points calculated
        :return:
            Plots the line graph
        """
        plt.figure()
        fig, ax = plt.subplots()
        # This locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    def train(self, pairs, vocab_size, n_iters, is_attention=False, hidden_size=256,
              print_every=1000, plot_every=100, learning_rate=0.01):
        """
            Trains the input and output with seq2seq or attention architecture

        :param pairs: list
            Pairs of inputs and corresponding outputs
        :param vocab_size: int
            Size of the vocabulary
        :param n_iters: int
            Number of iterations of training
        :param is_attention: bool
            If true trains on Attention architecture, else trains on Seq2Seq architecture
        :param hidden_size: int
            Size of the hidden nodes of the encoder and decoder
        :param print_every: int
            Iteration for printing periodically
        :param plot_every: int
            Iteration to plot the loss periodically
        :param learning_rate: float
            Learning rate
        :return: None
            Prints and plots the loss periodically
        """
        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        encoder = EncoderRNN(vocab_size, hidden_size).to(device)
        seq2seq_decoder = DecoderRNN(hidden_size, vocab_size).to(device)
        attn_decoder = AttnDecoderRNN(hidden_size, vocab_size, dropout_p=0.1).to(device)

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(seq2seq_decoder.parameters(),
                                      lr=learning_rate)  # Same optimizer for both decoders

        # TODO : The following is preprocessing step. Have to create pairs of inputs and outputs
        training_pairs = [tensors_from_pairs(random.choice(pairs)) for i in range(n_iters)]

        criterion = nn.NLLLoss()

        training_arch = TrainArch(encoder, encoder_optimizer, decoder_optimizer, criterion, max_length=100)

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            if not is_attention:
                loss = training_arch.seq2seq(input_tensor, target_tensor, seq2seq_decoder)
                print_loss_total += loss
                plot_loss_total += loss

            else:
                loss = training_arch.attention(input_tensor, target_tensor, attn_decoder)
                print_loss_total += loss
                plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d  %d%%) %.4f' % (self.__time_since(start, iter / n_iters), iter,
                                              iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        self.__show_plot(plot_losses)
