# -*- coding: utf-8 -*-
# pylint: disable=consider-iterating-dictionary

''' Get bert embedding '''

import tensorflow as  tf
import os
import collections
import  six
import numpy  as np
import json
from .tokenization import FullTokenizer
from .modeling import BertConfig, BertModel
    
# This is your Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

flags = tf.flags
 
FLAGS = flags.FLAGS
 
BERT_PATH = os.path.join(ROOT_DIR, 'chinese_L-12_H-768_A-12')
 
flags.DEFINE_string(
    "bert_config_file", os.path.join(BERT_PATH, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)
 
flags.DEFINE_string(
    "vocab_file", os.path.join(BERT_PATH, 'vocab.txt'),
    "The config vocab file"
)
 
flags.DEFINE_string(
    "init_checkpoint", os.path.join(BERT_PATH, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model)."
)


def convert_single_example(vectors, maxlen=10):
    length=len(vectors)
    if length>=maxlen:
        return  vectors[0:maxlen], [1]*maxlen, [0]*maxlen
    else:
        input=vectors+[0]*(maxlen-length)
        mask=[1]*length+[0]*(maxlen-length)
        segment=[0]*maxlen
        return input, mask, segment



def get_words_representation(word_list):
    """ return the embedding of the request text list
        Support Chinese words.
        Pool the tokens vectors to become word representation
    """
    tokenizer = FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)
    
    init_checkpoint = FLAGS.init_checkpoint
    use_tpu = False
    
    sess = tf.Session()
    
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    
    print(init_checkpoint)
    
    is_training=False
    use_one_hot_embeddings=False

    input_ids_p   = tf.placeholder(shape=[None,None],dtype=tf.int32,name="input_ids_p")
    input_mask_p  = tf.placeholder(shape=[None,None],dtype=tf.int32,name="input_mask_p")
    segment_ids_p = tf.placeholder(shape=[None,None],dtype=tf.int32,name="segment_ids_p")
    
    model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids_p,
            input_mask=input_mask_p,
            token_type_ids=segment_ids_p,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
    
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, init_checkpoint)
    #####################################################################################
    word2vec = {}      
    # mark the segment of each word    
    n = 150
    chunks_list = [word_list[i:i + n] for i in range(0, len(word_list), n)] 
    for chunks in chunks_list:
        segments = {}
        start = 0
        end = 0
        concat_indice = [tokenizer.vocab.get("[CLS]")]  
        for word in chunks:
            start = end + 1
            tokens = [tokenizer.vocab.get(token) for token in tokenizer.tokenize(word)]
            tokens += [tokenizer.vocab.get("[SEP]")]
            concat_indice += tokens
            end = len(concat_indice) # always mark the "[SEP]" as boundary
            segments[word] = (start, end)
        assert(len(segments) == len(chunks))

        input, mask, segment = convert_single_example(concat_indice, 
                maxlen=len(concat_indice))
        input_ids      = np.reshape(np.array(input), [1, -1])
        input_mask     = np.reshape(np.array(mask), [1, -1])
        segment_ids    = np.reshape(np.array(segment), [1, -1])
        embeddings     = tf.squeeze(model.get_sequence_output())
        representations = sess.run(embeddings, 
            feed_dict={"input_ids_p:0":input_ids, "input_mask_p:0":input_mask, 
                "segment_ids_p:0":segment_ids})
        representations = np.array(representations)
        # pool out each word
        for word, (start, end) in segments.items():
            word_rep = np.mean(representations[start:end], axis=0)
            word2vec[word] = word_rep
    
    return word2vec


def pool_vectors(encoder_layer, input_mask, pooling_strategy='REDUCE_MEAN'):
  minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
  mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
  masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
  masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
          tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
  if pooling_strategy == 'REDUCE_MEAN':
    pooled = masked_reduce_mean(encoder_layer, input_mask)
  elif pooling_strategy == 'REDUCE_MAX':
    pooled = masked_reduce_max(encoder_layer, input_mask)
  elif pooling_strategy == 'REDUCE_MEAN_MAX':
    pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                        masked_reduce_max(encoder_layer, input_mask)], axis=1)
  return pooled

 
if __name__ == "__main__":
    # server = pywsgi.WSGIServer(('0.0.0.0', 19877), app)
    # server.serve_forever()
    pass