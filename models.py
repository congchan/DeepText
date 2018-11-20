#! -*- coding: utf-8 -*-
# pylint: disable=consider-iterating-dictionary

""" Models """
from __future__ import absolute_import
from __future__ import print_function

__all__ = [ 
            'text_rnn',
            ]

from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import *
from keras.layers import Input,concatenate, merge, TimeDistributed, Bidirectional, GlobalMaxPooling1D, MaxPool1D
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


def text_rnn(max_length, emb_size, max_words, class_num, pre_train_emb=None):
    input = Input(shape=(max_length,), dtype='int32', name='input')

    embeddings_initializer = 'uniform'
    if pre_train_emb is not None:
        embeddings_initializer = initializers.Constant(pre_train_emb)
    embed_input = Embedding(output_dim=emb_size, dtype='float32', input_dim=max_words + 1, 
                            input_length=max_length, 
                            embeddings_initializer=embeddings_initializer,
                            )(input)


    drop_out_input = Dropout(0.5, name='dropout_word')(embed_input)
    bi_layer = Bidirectional(GRU(128, return_sequences=True))(drop_out_input)
    output = Dense(class_num, activation='softmax', name='output')(Flatten()(bi_layer))

    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
