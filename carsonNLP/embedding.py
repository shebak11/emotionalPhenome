from io import open
import unicodedata
import string
import re
import os
import random
import pickle
import numpy as np

# GLOVE Vectors
import torchtext.vocab as vocab

class Vocabulary:
    
    def __init__(self):
        
        '''
        Use this class to incorpporate a folder of text documents into a vocabulary where the
        words in the text and the punctuation including ! ,  ? etc are converted to GloVe vectors
        and the words/tokens in the text documents not in glove are initialized as random vectors
        Store the string-token-to-index-token-mapping in the word2index and index2word dictionaries. 
        '''
        
        self.trimmed = False # gets changed to True first time Lang.trim(min_count) is called
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.next_empty_index = len(self.index2word) # Count <structure> tokens
        self.num_nonwordtokens = len(self.index2word) # nonwordtokenslike <UNK>
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
            
    def normalize_string(self, s):
        '''this scanbe updated, since glove collapses  certain punctuation like ' and ` '''

        #s = self.unicode_to_ascii(s.lower().strip())
        s = self.unicode_to_ascii(s.lower())
        #s = s.replace("'s"," 's")
        #s = s.replace("'"," ' ")
        #s = s.replace("`"," ` ")
        s = s.replace(" utc "," ")
        s = s.replace(" utcfriday "," ")
        s = s.replace(" selfpost "," ")
        #s = re.sub(r"[^a-zA-Z0-9.!?,'`\"]+", r" ", s)
        s = re.sub(r"[^a-zA-Z0-9,.!? \"]+", r"", s)
        s = re.sub(r"([,.!? \"])", r" \1 ", s)
        s = ' '.join(s.split())
        return s
        
    def index_word(self, word):
        ''' 
        Updates word2index, index2word, word2count one string (word) at a time. 
        
        NOTE: this function does not normalize strings, for example if you want the 
        uppercase and lowercase of a word to be counted as the same word, you have to do that
        with another function'''
        
        if word not in self.word2index:
            self.word2index[word] =  self.next_empty_index
            self.index2word[self.next_empty_index] = word
            self.next_empty_index += 1
            self.word2count[word] = 1
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
        self.next_empty_index = len(self.index2word) # Count <structure> tokens
        self.num_nonwordtokens = len(self.index2word)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

        for word in keep_words:
            self.index_word(word)
            
    def index_sentence(self, sentence):
        '''
        Absorbs a sentence string into the token dictionary one word at a time by using ' ' as 
        a delimiter and using the index_word function which updates word2index, index2word, and
        word2count one string (word) at a time.
        '''
        for word in sentence.split(' '):
            self.index_word(word)
    
    def makeEmbedding(self, min_word_count, glove, path_to_folder_of_text, embed_dim = 50):
        
        '''
        min_word_count: integer
        glove: torchtext.vocab.GloVe() object or None to intantiate within this class
        path_to_folder_of_text: directory/folder inside which contains the document or documents
        that will be used to generate the vocabulary
        embed_dim:integer, number of dimensions you wish to represent words/tokens with
        '''
        
        for file in os.listdir(path_to_folder_of_text):
            lines = open(os.path.join(path_to_folder_of_text,file)).read().strip().split('\n')

            for sentence in lines:
                self.index_sentence(self.normalize_string(sentence))
    
        print("trimming...")    
        self.trim(min_word_count)
        
        embed_dim = glove.vectors.size(1)
                    
        print("building embedding from glove...")
        
        embedding = np.zeros((len(self.index2word), embed_dim)).astype(np.float32)
        
        # GloVe does not have <structure> tokens so we will randomy initialize these
        for i in range(self.num_nonwordtokens):
            embedding[i,:] = np.random.uniform(-0.5,0.5,embed_dim).astype(np.float32)
            
        # the remaining tokens in index2word will be randomly or pre-initialized  
        for i in range(self.num_nonwordtokens,len(self.index2word)):
            
            if self.index2word[i] in glove.stoi:
                embedding[i,:] = glove.vectors[glove.stoi[self.index2word[i]]]
            else:
                embedding[i,:] = np.random.uniform(-0.5,0.5,embed_dim).astype(np.float32)
                
        print('finished embeddings')
        return self.index2word, self.word2index, embedding 