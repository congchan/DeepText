# -*- coding: utf-8 -*-
# pylint: disable=consider-iterating-dictionary


'''  数据处理  '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [ 'DataProcessor',
            'SampleProcessor',
            ]

import json, csv, logging
import os, sys
import nltk
from nltk.util import ngrams
import jieba
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import collections
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import bert.bert_embedding as bert_embedding


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
        with open(input_file, "r", encoding='utf-8') as f:
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

    @staticmethod
    def write_file(data, out_file):
        " write data to out_file "
        with open(out_file, 'w', encoding='utf-8') as the_file:
            for line in data:
                the_file.write(line+'\n')

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

    @staticmethod
    def load_bert_embedding(vob_size, emb_size, word2id):
        """ Get bert pre-trained representation,  
            for example, pre-trained chinese_L-12_H-768_A-12, 
                the hidden_size is 768 
        """        
        num_words = min(vob_size, len(word2id)) + 1
        rep_matrix = np.zeros((num_words, emb_size))
        word2vec = bert_embedding.get_words_representation(list(word2id))
        for word, i in word2id.items():
            if i > vob_size:
                continue
            rep_matrix[i] = word2vec[word]
        return rep_matrix


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

    
    
class SampleProcessor(DataProcessor):
    """ Sample processor for the classification data set.
        Tranform the text to tensor for training         
        if use pre-train model, need vocabulary file
        usage:
            process data files 
            >>> processer = SampleProcessor(config, )

            provide your own data in list format [train_X, train_Y, test_X, test_Y]
            >>> processer = SampleProcessor(config, data)

    """
    def __init__(self, config, data=None,): 
        self.config = config
        self.init_checkpoint = config.get('init_checkpoint', None)
        self.init_checkpoint_config = config.get('init_checkpoint_config', None)
        self.vocab_file = config.get('vocab_file', None)
        self.max_seq_length = config.get('max_seq_length', 500)
        self.vob_size = config.get('vob_size', 10000)
        self.ngram = config.get('ngram', 1)
        self.output_dir = config.get('output_dir', './output/')
        self.which_model = config.get('which_model', None)
        self.class_weight = None
        self.class_num = None

        if data is None:
            logging.info("="*10+" start processing data from files: \n {}".format(
                            [config.get('train_file'), config.get('test_file')]))
            self.train_X, self.train_Y, self.test_X, self.test_Y = self.encode_data(
                    *self.get_data_from_file(config.get('train_file')), 
                    *self.get_data_from_file(config.get('test_file')), 
                    )
            logging.info("="*10+" Completed processing data "+"="*10)
        else:
            self.train_X, self.train_Y, self.test_X, self.test_Y = data

    @staticmethod
    def _tokenize(text):
        """ define your tokenizer, return text that being tokenized """
        return " ".join(jieba.lcut(text))
        

    def get_data_from_file(self, input_file, set_type=None):
        """ 读取文件数据, 提取label. """
        lines = self.read_tsv(input_file)
        X = []
        Y = []
        for line in lines[1:]:  
            sequence = line[-1]
            label = line[1]
            X.append(self._tokenize(sequence))
            if set_type != 'predict':
                Y.append(label)

        return X, Y

    def get_word2id(self, vocab):
        " return word2id mapping "
        word2id = {}
        for i, v in enumerate(vocab):
            word2id[v] = i
        return word2id


    def encode_data(self, train_X, train_Y, test_X, test_Y):
        self.word2id, self.label2id, self.class_weight, _ = self._encode_mapping(train_X, train_Y)
        train_X = self._text_to_sequence(self.word2id, train_X, self.ngram, self.max_seq_length)
        train_Y = self._label_to_onehot(train_Y, self.label2id)

        test_X = self._text_to_sequence(self.word2id, test_X, self.ngram, self.max_seq_length)
        test_Y = self._label_to_onehot(test_Y, self.label2id)

        return train_X, train_Y, test_X, test_Y
 
    def _encode_mapping(self, train_X, train_Y):
        if self.init_checkpoint and self.init_checkpoint_config and self.vocab_file:
            word2id = self.get_vocab(self.read_text(self.vocab_file))
            label2id, class_weight, parameter = json.load(
                open(self.init_checkpoint_config, 'r', encoding='utf-8'))
        else:
            # build vocabulary and word2id mapping
            if self.which_model != 'fasttext':
                self.ngram = 1
            vectorizer = CountVectorizer(token_pattern = '[^\s]+', 
                                        ngram_range=(1, self.ngram), 
                                        max_df=1.0, 
                                        min_df=1, 
                                        max_features=self.vob_size)
            vectorizer.fit(train_X)
            vocab = list(vectorizer.vocabulary_.keys())
            word2id = self.get_word2id(vocab)
            
            # create label2id mapping
            Binarizer = MultiLabelBinarizer()
            Binarizer.fit_transform([train_Y])
            label2id = {label: index for index, label in enumerate(Binarizer.classes_)}
            self.class_num = len(Binarizer.classes_)
            # create class_weight for unbalanced data during training
            # label_freq = nltk.FreqDist(train_Y)
            label_freq = collections.Counter(train_Y)
            logging.info('label_freq statistics: {}'.format(label_freq))
            class_weight = {int(index): max(label_freq.values())/label_freq[label] 
                            for index, label in enumerate(Binarizer.classes_)}
            
            # store necessary model parameter
            parameter = {'model':self.which_model, 
                        'max_length':self.max_seq_length, 
                        'ngram':self.ngram}
            
            # save vocab, label2id, class_weight, parameter
            self.write_file(vocab, os.path.join(self.output_dir, 'vocab.txt'))
            json.dump([label2id, class_weight, parameter], 
                    open(os.path.join(self.output_dir, 'model_config.json'), 
                            'w', encoding='utf-8'), ensure_ascii=False)
        
        return word2id, label2id, class_weight, parameter


