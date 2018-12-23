# -*- coding: utf-8 -*-
# pylint: disable=consider-iterating-dictionary

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

''' nlp task '''

import configparser
import pandas as pd
import os
import sys
import logging
sys.path.append(os.getcwd())

import re
import json
import jieba
import time
import numpy as np
import nltk
from nltk import FreqDist
from nltk.util import ngrams
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt 

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from models import * 
from pathlib import Path
from data_process import *

class Task(object):
    """ Default task is classification task,
        default model is fasttest, as a baseline.
        Overwrite/ add any methods you want in your model. 
    """
    def __init__(self, **kwargs):
        self.model_list = { 
                            'fasttext': fasttext, 
                            'textcnn': text_cnn, 
                            'textrcnn': text_rcnn,
                            'textinception': text_inception,
                            'textrnn': text_rnn,
                            'textrnnatt': text_rnn_attention,
                            }
        self.model = self.model_list[kwargs.get('which_model', 'fasttext')]
        self.output_dir = kwargs.get('output_dir', './output/')
        self.which_model = kwargs.get('which_model', 'fasttext')

    def train(self, batch_size, epochs, 
            x_train, y_train, x_dev, y_dev,
            model=None, class_weight=None):
            
        start_time = time.time()
        checkpoint = os.path.join(self.output_dir, 
                self.which_model+'_e{epoch:02d}_vacc{val_acc:.02f}.h5')
        cvs_log = os.path.join(self.output_dir, 
                self.which_model+'.log')
        if model is None:
            model = self.model
        history  = model.fit([x_train], y_train,
                validation_data=(x_dev, y_dev),
                batch_size=batch_size, 
                epochs=epochs, 
                class_weight=class_weight, 
                shuffle=False,
                verbose=1,
                callbacks=[ keras.callbacks.ModelCheckpoint(checkpoint,
                                monitor='val_acc', 
                                verbose=0, 
                                save_best_only=True,
                                period=5),
                            keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=5, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None, 
                                restore_best_weights=False),
                            # keras.callbacks.RemoteMonitor(root='http://localhost:9000', 
                            # # Events are sent to root + '/publish/epoch/end/'
                            #     path='/publish/epoch/end/', 
                            #     field='data', 
                            #     headers=None, 
                            #     send_as_json=False),
                            keras.callbacks.CSVLogger(cvs_log, separator=',', 
                                append=False),
                            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1, 
                                patience=3, 
                                verbose=0, 
                                mode='auto', 
                                min_delta=0.0001, 
                                cooldown=0, 
                                min_lr=0)
                            ],
                )
        end_time = time.time()
        print("Complet training in: {:.02f}s".format(end_time - start_time))
        return history

    def eval(self, model, sequence, true_label, batch=8):
        acc = model.evaluate(x=sequence, y=true_label, batch_size=batch, verbose=1, 
                sample_weight=None, steps=None)
        return acc
        
    def predict(self, model, sequence, id2label, batch=8):
        res = []
        temp = model.predict([sequence], batch_size=batch)
        for i in temp:
            res.append(id2label[str(np.argmax(i,axis=0))])
        return res

    def plot_training_history(self, history, to_file):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

class Classifier(Task):
    """
    Define a classifier for classification task, inherited from Task.
    
    Available models
            'fasttext', 
            'textCNN', 
            'textRCNN',
            'textInception',
            'textRNN',
            'textRNNATT',
    
    You could define your new model in models.py, and added in self.model_list

    Data：make use of SampleProcessor 
    -----------------------------------------------------------------------------------
    usage:
        init a new classifier:
        >>> myclassifier = Classifier(max_seq_length=500, emb_size=300)
        
        init a new classifier with config parameters:
        >>> myclassifier = Classifier(**config)

        load from a check point: 
        >>> myclassifier = Classifier(check_point = './pre_trained/fasttext.h5')

    """
    def __init__(self, **kwargs):        
        super().__init__()

        self.check_point = kwargs.get('check_point', None)
        self.check_point_config = kwargs.get('check_point_config', None)
        self.output_dir = kwargs.get('output_dir', './output/')
        self.max_seq_length = kwargs.get('max_seq_length', 512)
        self.emb_size = kwargs.get('emb_size', 128)
        self.vob_size = kwargs.get('vob_size', 10000)
        self.class_num = kwargs.get('class_num', 10)
        self.which_model = kwargs.get('which_model', " ").lower()
        self.pre_train_emb = kwargs.get('pre_train_emb', None)

        self.model = self.build_model()

    def build_model(self):
        """ load a pre-trained model, or build a new model """
        if self.check_point:
            model = keras.models.load_model(self.check_point)
            print("Load model from {}".format(self.check_point))

        elif self.which_model and self.which_model in self.model_list:
            model = self.model_list[self.which_model](
                    self.max_seq_length, 
                    self.emb_size, 
                    self.vob_size, 
                    self.class_num,
                    self.pre_train_emb)
            print("Init a new {} model".format(self.which_model))

        else:
            error_msg  = 'Please specify a valid "which_model" value from {}.'.format(
                    self.model_list.keys())
            error_msg += 'Or provide a valid pretrained model file'
            raise Exception(error_msg) 

        return model


import unittest
class TestStringMethods(unittest.TestCase):

    def test(self):
        config ={'max_seq_length': 64,
            'vob_size': 10000,
            'emb_size': 128,
            'num_train_epochs': 2,
            'train_batch_size': 512,
            'if_train': True,
            'which_model': 'textRNN',
            'train_file': './data/train.tsv',
            'dev_file': "./data/dev.tsv",
            'test_file': "./data/test.tsv",
            'output_dir': './train',            
            }

        Path(config['output_dir']).mkdir(exist_ok=True)
        log_path = os.path.join(config['output_dir'], "train.log")

        logging.basicConfig(level=logging.INFO,
                            handlers = [
                                logging.StreamHandler(),
                                logging.FileHandler(log_path)
                            ])

        logging.info("Configuration: {}".format(config))

        processer = SampleProcessor(config, )

        ################################################################################
        # 提取中文预训练BERT模型的vocab向量表达，作为你的模型的embedding参数
        # Notice: pre-trained bert model embedding size should be the same as your model
        tic = time.time()
        
        pre_train_emb = processer.load_bert_embedding(
                processer.vob_size, config['emb_size'], processer.word2id)
        toc = time.time()
        logging.info("Cost {:.2f}s to load BERT representation for {} words.".format(
                toc-tic, config['vob_size']))
        config.update({"pre_train_emb": pre_train_emb})
        ################################################################################
        class_num = processer.class_num
        config.update({"class_num":class_num})

        classifier = Classifier(**config)
        history = classifier.train(batch_size=config['train_batch_size'], 
                epochs=config['num_train_epochs'], 
                x_train=processer.train_X, 
                y_train=processer.train_Y, 
                x_dev=processer.test_X, 
                y_dev=processer.test_Y,
                model=None, 
                class_weight=processer.class_weight)

        
        logging.info('Validation accuracry log:')
        for acc in history.history.get('val_acc', ""):
            logging.info(acc)



if __name__ == '__main__':
    " demo "
    unittest.main()
    conf = configparser.ConfigParser()
    conf.read('./config.ini')
    
