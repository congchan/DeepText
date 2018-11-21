#! -*- coding: utf-8 -*-
# pylint: disable=consider-iterating-dictionary

""" Models """
from __future__ import absolute_import
from __future__ import print_function

__all__ = [ 
            'text_rnn',
            'text_rnn_attention',
            'AttentionWithContext', 
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
    " Text RNN model using GRU cell"
    return _bilstm_attention(max_length, emb_size, max_words, class_num, False, pre_train_emb)

def text_rnn_attention(max_length, emb_size, max_words, class_num, pre_train_emb=None):
    " Text RNN model using GRU cell with attention mechanism"
    return _bilstm_attention(max_length, emb_size, max_words, class_num, True, pre_train_emb)

def _bilstm_attention(max_length, emb_size, max_words, class_num, attention=False, pre_train_emb=None):
    """ bidirectional lstm with simple multiply-attention mechanism,
        by default use GRU cell.
    """
    input = Input(shape=(max_length, ), dtype='int32', name='input')
    embeddings_initializer = 'uniform'
    if pre_train_emb is not None:
        embeddings_initializer = initializers.Constant(pre_train_emb)
    embed_input = Embedding(output_dim=emb_size, dtype='float32', input_dim=max_words + 1, 
                            input_length=max_length, 
                            embeddings_initializer=embeddings_initializer,
                            )(input)
    drop_out_input = Dropout(0.5, name='dropout_word')(embed_input)
    bi_layer = Bidirectional(GRU(128, dropout=0.1, return_sequences=True))(drop_out_input)
    
    if attention:
        bi_layer = AttentionWithContext()(bi_layer)
    else:
        bi_layer = Flatten()(bi_layer)

    output = Dense(class_num, activation='softmax', name='output')(bi_layer)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class AttentionWithContext(Layer):
    """ Attention operation, with a context/query vector, for temporal data.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
    """
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        super(AttentionWithContext, self).build(input_shape)

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        a = K.exp(uit)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    drop_out_input = Dropout(0.5, name='dropout_word')(embed_input)
    bi_layer = Bidirectional(GRU(128, return_sequences=True))(drop_out_input)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
