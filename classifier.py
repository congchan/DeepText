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
