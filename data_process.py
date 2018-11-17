# -*- coding: utf-8 -*-
# pylint: disable=consider-iterating-dictionary


'''  数据处理  '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['DataProcessor']

import json, csv, logging
import os, sys
import nltk
from nltk.util import ngrams
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import collections
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer



class DataProcessor(object):
    """ Base class for data converters for sequence classification data sets.
        helper funcitons [read_tsv, read_text, read_json]
    """ 
    @staticmethod
    def read_text(input_file):
        '''read data from text file'''
        text = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                text.append(line.strip())
        return text

    @staticmethod
    def read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
    @staticmethod
    def read_json(input_file):
        """Reads json file. """
        with open(input_file,'r',encoding='utf-8') as f:
            lines = json.loads(f.readline())
            return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        pass

    def create_examples(self, lines, set_type=None,):
        """ Retrieve the content and labels. 
            If prediction mode, ignore label
        """
        data = []
        for line in lines: 
            label, sequence = line.split(" ", 1)   
            if set_type == 'predict':
                data.append({'content': self._tokenize(sequence)})
            else:
                data.append({'content': self._tokenize(sequence), 'label': label})
        
        return data

    def _encode_data(self):
        """ encode the content into one-hot-vector,
            encode the labels into multi-label binary array
        """
        pass

    @staticmethod
    def _tokenize(text):
        """ define your tokenizer, return text that being tokenized """
        return text

    @staticmethod
    def load_embedding(pre_train_emb, emb_size, vob_size, word2id):
        """ demo prepare embedding matrix 
            list of pre_train_emb:
            {glove: glove.6B.100d.txt, 
            #TODO word2vec: gensim,
            #TODO fasttext: gensim,
            #TODO bert: remote service}
        """
        print('Indexing word vectors.')
        # glove emb file
        embeddings_index = {}
        with open(pre_train_emb) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))

        num_words = min(vob_size, len(word2id)) + 1
        embedding_matrix = np.zeros((num_words, emb_size))
        for word, i in word2id.items():
            if i > vob_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix



    def _pad_ngrams(self, text_list, ngram):
        " pad ngrams behind text "
        if ngram < 2:
            return text_list
        new_text_list = []
        print('Before padding, mean sequence length: {}'.format(
            np.mean(list(map(lambda x:len(x.split()), text_list)), dtype=int)))
        for content in text_list:
            new_content = []
            for n in range(ngram):
                new_content.extend([' '.join(p) for p in ngrams(content.split(), n+1)])
            new_text_list.append(new_content)
        print('After padding, mean sequence length: {}'.format(
            np.mean(list(map(lambda x:len(x), new_text_list)), dtype=int)))
        return new_text_list

    
    def _text_to_sequence(self, word2id, text_list, ngram, max_length):
        " convert text to sequence "
        vobs = word2id.keys()
        new_text_list = self._pad_ngrams(text_list, ngram)
        sequence_list = [[word2id[w] for w in content if w in vobs] 
                for content in new_text_list]
        sequence_list = pad_sequences(sequence_list, maxlen=max_length)
        return sequence_list

    
    def _label_to_onehot(self, label_list, label2id, set_type=None):
        " convert label to onehot format "
        if set_type == 'predict':
            label_list = None
        else:
            encoder = OneHotEncoder(categories='auto') 
            label_list = [[label2id[label]] for label in label_list]
            encoder.fit([[k] for k in label2id.values()])
            label_list = encoder.transform(label_list).toarray()
        
        return label_list

    
    