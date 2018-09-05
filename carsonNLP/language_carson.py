

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import time
import random
import pickle
import numpy as np
import math

import torch
from torch.nn import functional
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# GLOVE Vectors
import torchtext.vocab as vocab

#For Batching
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

use_cuda = torch.cuda.is_available()

THRESHOLD_FOR_UNIGRAM_OVERLAP = 0.8

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [lang.PAD_token for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size, pairs, lang):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexesFromSentence(lang, pair[0]))
        target_seqs.append(indexesFromSentence(lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if use_cuda:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        #hidden = self.hidden0.repeat(1, batch_size, 1)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N, view is like reshape()

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if use_cuda:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding,
                 num_layers = 3, bidirectional = False, train_embedding = True):

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        embedding = torch.from_numpy(embedding).float()
        if use_cuda:
            embedding.cuda()
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding, requires_grad=train_embedding)
        self.gru = nn.GRU(embedding.shape[1], hidden_size, num_layers, bidirectional=bidirectional)

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # make the initial hidden state learnable as well
        hidden0 = torch.zeros(self.num_layers*num_directions, 1, self.hidden_size)

        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input_seqs, input_lengths, hidden):

        if use_cuda:
            input_seqs.cuda()
        batch_size = input_seqs.size(1)
        hidden = self.hidden0.repeat(1, batch_size, 1)

        self.embedded = self.embedding(input_seqs)
        #self.packed = torch.nn.utils.rnn.pack_padded_sequence(self.embedded, input_lengths)
        #output, hidden = self.gru(self.packed, hidden)
        output, hidden = self.gru(self.embedded, hidden)
        #output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output) # unpack (back to padded)

        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:] # Sum bidirectional outputs

        # ouput (max_len x batch_size x hidden_size)
        # hidden ( n_layers * 2(if bidirectional) x batch_size x hidden_size )
        return output, hidden


    def initHidden(self):

        if use_cuda:
            return self.hidden0.cuda()
        else:
            return self.hidden0


def sentenceFromIndexes(lang, idx_array):
    '''
    converts a list of Variables that contain an index back to a sentence
    '''
    words = []
    for idx in idx_array:
        if idx.data[0] in lang.index2word:
            words.append(lang.index2word[idx.data[0]])
        else:
            words.append('<UNK>')
    return " ".join(words)

def indexesFromSentence(lang, sentence):
    '''
    account for strings not in the vocabulary by using the unknown token
    '''
    sentence_as_indices = []
    sentence = lang.normalize_string(sentence)
    for word in sentence.split(' '):
        if word in lang.word2index:
            sentence_as_indices.append(lang.word2index[word])
        else:
            sentence_as_indices.append(lang.UNK_token)

    sentence_as_indices.append(lang.EOS_token)

    return sentence_as_indices


def variableFromSentence(lang, sentence):
    '''
    add EOS token to sequence of idices and make a column vector
    '''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(lang.EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair,lang):
    input_variable = variableFromSentence(lang, pair[0])
    target_variable = variableFromSentence(lang, pair[1])
    return (input_variable, target_variable)

######## the pair indices are returned as 2 LongTensor Variables in torch #############


######## Tells you how long youve been training and how much longer you have left ####

import time
import math

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

######################################################################3

############### plot_losses #######################################

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

####################################################################

class Lang:

    def __init__(self, name):

        '''
        Store the string token to index token
        mapping in the word2index and index2word
        dictionaries.
        '''

        self.name = name
        self.trimmed = False # gets changed to True first time Lang.trim(min_count) is called
        self.word2index = {"<PAD>" : 0 ,  "<SOS>" : 1, "<EOS>" : 2 , "<UNK>" : 3}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = len(self.index2word) # Count default tokens
        self.num_nonwordtokens = len(self.index2word)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

    def index_sentence(self, sentence):
        '''
        Absorbs a sentence string into the token dictionary
        one word at a time using the index_word function
        increments the word count dictionary as well
        '''
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


    def trim(self, min_count):
        '''
        Removes words from our 3 dictionaries that
        are below a certain count threshold (min_count)
        '''
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = len(self.index2word) # Count default tokens
        self.num_nonwordtokens = len(self.index2word)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

        for word in keep_words:
            self.index_word(word)

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub("newlinechar", "", s)
        s = s.replace("'","")
        s = s.replace(".","")
        s = s.replace("n t ","nt ")
        s = s.replace("i m ","im ")
        s = s.replace("t s ","ts ")
        s = s.replace(" s ","s ")
        s = s.replace(" re "," are ")
        s = s.replace("i ve ","ive ")
        s = s.replace(" d ","d ")
        s = ' '.join(s.split())
        return s

    def filterPair(self, p, max_sent_len, min_sent_len):

        '''
        Your Preferences here
        '''

        return len(p[0].split(' ')) < max_sent_len and \
               len(p[1].split(' ')) < max_sent_len and \
               len(p[1].split(' ')) > min_sent_len and \
               len(p) == 2 and \
               "https://" not in p[1]


    def make_pairs(self, path_to_tab_sep_dialogue,
                   max_sent_len = 20, min_sent_len = 4):

        print("making final_pairs list ...")
        lines = open(path_to_tab_sep_dialogue).read().strip().split('\n')

        final_pairs = []
        i = 0
        for l in lines:

            pair = [self.normalize_string(sentence) for sentence in l.split('\t')]

            if self.filterPair(pair,max_sent_len, min_sent_len):

                filtered_pair = []

                for sentence in pair:

                    self.index_sentence(sentence)
                    filtered_pair.append(sentence)

                final_pairs.append(filtered_pair)
        print("number of pairs", len(final_pairs))
        return final_pairs

    def tokens2glove(self, min_word_count,glove, mbed_dim = 50):

        print("trimming...")
        self.trim(min_word_count)

        if glove is None:
            glove = vocab.GloVe(name='6B', dim=embed_dim)
            print('Loaded {} words'.format(len(glove.itos)))
        else:
            embed_dim = glove.vectors.size(1)

        print("building embedding from glove...")
        embedding = np.zeros((len(self.index2word), embed_dim)).astype(np.float32)
        for i in range(self.num_nonwordtokens):
            embedding[i,:] = np.random.uniform(-1,1,embed_dim).astype(np.float32)
        for i in range(self.num_nonwordtokens,len(self.index2word)):
            if self.index2word[i] in glove.stoi:
                embedding[i,:] = glove.vectors[glove.stoi[self.index2word[i]]]
            else:
                embedding[i,:] = np.random.uniform(-1,1,embed_dim).astype(np.float32)

        return self.index2word, self.word2index, embedding, self.n_words #torch.from_numpy(embeddings).float()

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):

    if torch.cuda.is_available():
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def losses_masks_cross_entropy(logits, target, length):

    if torch.cuda.is_available():
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    #loss = losses.sum() / length.float().sum()
    return losses, mask.float()

def indexes2string(seq_idx, lang):
    decoded_words = []
    for idx in seq_idx:
            decoded_words.append(lang.index2word[idx])
    return ' '.join(decoded_words)


# holds a candidate to be used in beam search.
# we need to store the current log_prob, hidden_state, decoded_words,

logSoftMaxFunc = nn.LogSoftmax()
realSoftMax = nn.Softmax()

class BeamSearchCandidate:

    # should start using [SOS]
    def __init__(self, lang, encoder_outputs, hidden_state, next_word_idx, log_prob, decoded_words, is_eos):
        self.encoder_outputs = encoder_outputs
        self.hidden_state = hidden_state
        self.seq_log_prob = log_prob # init
        self.decoded_seq_idx = decoded_words
        self.next_word_idx = next_word_idx
        self.is_eos = is_eos
        self.lang = lang

    # return a list of BeamSearchCandidate by feeding next word into rnn_decoder
    def feed_word_and_get_new_candidates(self, rnn_decoder, beam_width):

        # if you've already reached the EOS, don't return anymore
        # todo: add some parameters for beam search to fix the length problem
        if (self.is_eos):
            #print("Got EOS output for this candidate already...not creating more candidates")
            return [self]

        else:

            decoder_input = Variable(torch.LongTensor([self.next_word_idx]))

            if use_cuda:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden, decoder_attention = rnn_decoder(
                decoder_input, self.hidden_state, self.encoder_outputs)

            log_probs = logSoftMaxFunc(decoder_output)

            top_log_probs, top_i = log_probs.data.topk(beam_width)

            new_candidates = []
            for b in range(0, beam_width):
                curr_word = top_i[0][b]
                curr_word_log_prob = top_log_probs[0][b]
                is_eos_tmp = False
                if (curr_word == self.lang.EOS_token):
                    is_eos_tmp = True
                # update log prob
                new_log_prob = self.seq_log_prob + curr_word_log_prob
                # update word seq
                new_decoded_word_seq = list(self.decoded_seq_idx)
                new_decoded_word_seq.append(curr_word)
                # create new candidate and append to list
                new_beam_search_candidate = BeamSearchCandidate(self.lang, self.encoder_outputs, decoder_hidden, curr_word, new_log_prob, new_decoded_word_seq, is_eos_tmp)
                new_candidates.append(new_beam_search_candidate)

            return new_candidates

    # return the decoded sequence along with it's probability
    # could just keep it as log_prob since this is probably going to be really small
    def get_decoded_words_and_prob(self):
        decoded_words = []
        for idx in self.decoded_seq_idx:
            decoded_words.append(self.lang.index2word[idx])
        return (decoded_words, np.exp(self.seq_log_prob))

def beam_response(input_seq, encoder, decoder, lang, max_length):

    input_seqs = []

    #print(input_seq)
    input_seqs.append(indexesFromSentence(lang, input_seq))

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)

    if use_cuda:
        input_var = input_var.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_lengths, None)
    #encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    #input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([lang.SOS_token]))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    # Move new Variables to CUDA
    if use_cuda:
        decoder_input = decoder_input.cuda()

    ################# define beam search constants ######################

    beam_width = 3  # how many words to explore at each stage
    max_candidates = 10 # max number of sequences you want to consider at one time
    num_iterations = 10 # this should really be equal to the max length ???
    num_final_candidates = 5 # number of candidates you actually want to return

    ################ start beam search ##############################

    # initially we feed in SOS, we have no words in output so far, and log prob is 1.0
    starting_candidate = BeamSearchCandidate(lang, encoder_outputs, decoder_hidden,
                                             lang.SOS_token, 1.0, [], False)
    curr_candidates = [starting_candidate]

    for i in range(0, num_iterations):

        candidates_for_next_iteration = []
        for can in curr_candidates:
            next_candidates = can.feed_word_and_get_new_candidates(decoder, beam_width)
            # todo: does python have flatMap(...) like scala? would be way easier
            for next_can in next_candidates:
                candidates_for_next_iteration.append(next_can)
        curr_candidates = candidates_for_next_iteration
        if (len(curr_candidates) > max_candidates):
            # prune the list of candidates down, sort by log prob of sequence
            # could be sped up using a priority queue or something ?
            #print("We have {} candidates but the max is {}. Only keeping the top {} candidates".format(len(curr_candidates), max_candidates, max_candidates))
            curr_candidates = sorted(curr_candidates, key = lambda can : can.seq_log_prob, reverse=True)[0:max_candidates]

    # sort by log prob once more for final output
    final_candidates = sorted(curr_candidates, key = lambda can : can.seq_log_prob, reverse=True)[0:num_final_candidates]

    #for can in final_candidates:
        #print(can.get_decoded_words_and_prob())

    candidate_list = [can.get_decoded_words_and_prob() for can in final_candidates]
    sorted_candidate_list = sorted(candidate_list, key=lambda x: x[1])


    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    output_str_list, prob = sorted_candidate_list[-1]
    # decoder_hidden = sentence_encoding # Use last (forward) hidden state from encoder
    return output_str_list, prob, decoder_hidden

def string_cos_similarity(string1,string2, encoder, decoder, lang, max_length):

    response1_str_list, input_prob, input_encoding = beam_response(string1, encoder,
                                                     decoder, lang, max_length=max_length)
    response2_str_list, output_prob, output_encoding = beam_response(string2, encoder,
                                                       decoder, lang, max_length=max_length)
    dot = torch.dot(input_encoding.view(-1), output_encoding.view(-1))
    magprod = (torch.norm(input_encoding.view(-1))*torch.norm(output_encoding.view(-1)))

    cos_sim = dot/magprod

    return cos_sim

# get the log probability of output_seq given input_seq (assumes EOS is appened to it)
# input_seq is a list of indexes and so is output seq (both Variables?)
# returns: the log prob of output given input
def get_log_prob_given_input_and_output(lang, input_seq, output_seq, encoder, decoder):

    ###GPU VERSION

    if use_cuda:
        input_seq = input_seq.cuda()

    # Run input words through encoder (since we only have one senquence we don't need to do padding)
    encoder_outputs, encoder_hidden = encoder(input_seq, len(input_seq), None)

    # Get encoder hidden state to initially feed to decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder


    output_seq_log_prob = 0.0 # p = 1 ==> log(p) = 0

    # first "word" you feed into encoder is SOS token
    curr_word = Variable(torch.LongTensor([lang.SOS_token]))

    if use_cuda:
        curr_word = curr_word.cuda()

    # feed curr_word into the decoder and get the probability of predicting next_word. Then, set next_word = curr_word
    # and feed that into the decoder...
    for next_word_idx in output_seq:

        decoder_output, decoder_hidden, decoder_attention = decoder(
            curr_word, decoder_hidden, encoder_outputs)

        log_probs_over_vocab = logSoftMaxFunc(decoder_output)

        if use_cuda:
            next_word_idx = next_word_idx.cuda()

        #print("next_word_idx.data",next_word_idx.data)
        output_seq_log_prob += log_probs_over_vocab.data[0][next_word_idx.data]
        curr_word = next_word_idx



    return output_seq_log_prob

def chat_using_beam_search(input_seq, lang, encoder, decoder, max_length):

    #input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)
    #input_lengths = [len(input_seq)]
    #input_seqs = [indexesFromSentence(lang, input_seq)]
    #input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
    #[torch.cuda.LongTensor of size 9x1 (GPU 0)]

    input_seqs = []

    #print(input_seq)
    input_seqs.append(indexesFromSentence(lang, input_seq))

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)

    if use_cuda:
        input_var = input_var.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_lengths, None)
    #encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    #input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([lang.SOS_token]))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    # Move new Variables to CUDA
    if use_cuda:
        decoder_input = decoder_input.cuda()

    ################# define beam search constants ######################

    beam_width = 3  # how many words to explore at each stage
    max_candidates = 10 # max number of sequences you want to consider at one time
    num_iterations = 10 # this should really be equal to the max length ???
    num_final_candidates = 5 # number of candidates you actually want to return

    ################ start beam search ##############################

    # initially we feed in SOS, we have no words in output so far, and log prob is 1.0
    starting_candidate = BeamSearchCandidate(lang, encoder_outputs, decoder_hidden,
                                             lang.SOS_token, 1.0, [], False)
    curr_candidates = [starting_candidate]

    for i in range(0, num_iterations):

        candidates_for_next_iteration = []
        for can in curr_candidates:
            next_candidates = can.feed_word_and_get_new_candidates(decoder, beam_width)
            # todo: does python have flatMap(...) like scala? would be way easier
            for next_can in next_candidates:
                candidates_for_next_iteration.append(next_can)
        curr_candidates = candidates_for_next_iteration
        if (len(curr_candidates) > max_candidates):
            # prune the list of candidates down, sort by log prob of sequence
            # could be sped up using a priority queue or something ?
            #print("We have {} candidates but the max is {}. Only keeping the top {} candidates".format(len(curr_candidates), max_candidates, max_candidates))
            curr_candidates = sorted(curr_candidates, key = lambda can : can.seq_log_prob, reverse=True)[0:max_candidates]

    # sort by log prob once more for final output
    final_candidates = sorted(curr_candidates, key = lambda can : can.seq_log_prob, reverse=True)[0:num_final_candidates]

    #for can in final_candidates:
        #print(can.get_decoded_words_and_prob())


    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return final_candidates

def split_dialogue(dialog_seq):
	responses_from_A = []
	responses_from_B = []
	for i in range(0, len(dialog_seq)):
		if i % 2 == 0:
			responses_from_A.append(dialog_seq[i])
		else:
			responses_from_B.append(dialog_seq[i])
	return (responses_from_A, responses_from_B)


# remove "A: " or "B: " from response along with punctuation (using myString.translate(...) trick )
def clean_response_helper(response):
	a_idx = response.find("A: ")
	b_idx = response.find("B: ")
	if a_idx != -1:
		return response[a_idx + 3:].translate(str.maketrans('','',string.punctuation))
	if b_idx != -1:
		return response[b_idx + 3:].translate(str.maketrans('','',string.punctuation))
	return response.translate(None, string.punctuation)


def unigram_overlap_check(prev_response, next_response):
	words_prev_respose = set(prev_response.split(" "))
	words_next_response = set(next_response.split(" "))
	union = words_next_response.intersection(words_prev_respose)
	# check if more than 80% of the words in the next response overlap with the previous response
	if len(union)*1.0/len(words_prev_respose) > THRESHOLD_FOR_UNIGRAM_OVERLAP:
		return True
	else:
		return False

def check_termination_conditions(prev_response, next_response, dull_set):
	return prev_response in dull_set or unigram_overlap_check(prev_response, next_response)


# unigram diveristy ratio is the total # of words generated in ALL responses by an agent in a given conversation vs.
# the total number of unique words generated
def calculate_unigram_diversity_ratio(all_responses):
	unique_words = set()
	total_num_tokens = 0
	for response in all_responses:
		cleaned_response_split = clean_response_helper(response).split(" ")
		unique_words = unique_words.union(cleaned_response_split)
		total_num_tokens += len(cleaned_response_split)

	return len(unique_words) * 1.0 / total_num_tokens


# given a list of [x1,x2,x3...] generate a new list of all possible bigrams [(x1,x2), (x2,x3)..(x_{n-1}, x_{n})]
def generate_bigrams(some_list):
	bigram_list = []
	for i in range(0, len(some_list) - 1):
		bigram_list.append((some_list[i], some_list[i+1]))
	return bigram_list

def calculate_bigram_diveristy_ratio(all_responses):
	unique_words = set()
	total_num_tokens = 0
	for response in all_responses:
		cleaned_response_split_bigrams = generate_bigrams(clean_response_helper(response).split(" "))
		unique_words = unique_words.union(cleaned_response_split_bigrams)
		total_num_tokens += len(cleaned_response_split_bigrams)

	return len(unique_words) * 1.0 / total_num_tokens


def calculate_dialog_length(all_agent_responses,dull_set):
	prev_response = all_agent_responses[0]
	turn_counter = 0
	for response in all_agent_responses[1:]:
		next_response = response
		if check_termination_conditions(clean_response_helper(prev_response),
                                        clean_response_helper(next_response),dull_set):
			break
		turn_counter += 1
	return turn_counter
