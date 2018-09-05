

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

def pad_seq(seq, max_length, vocabClass):
    seq += [vocabClass.PAD_token for i in range(max_length - len(seq))]
    return seq

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
    add EOS token at the end
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

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

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


