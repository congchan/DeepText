# -*- coding: utf-8 -*-
# pylint: disable=consider-iterating-dictionary

''' Get bert embedding '''

import tensorflow as  tf
import os
import collections
import  six
# from gevent import monkey
# monkey.patch_all()
# from flask import Flask, request
# from gevent import pywsgi
import numpy  as np
import json
from .tokenization import FullTokenizer
from .modeling import BertConfig, BertModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

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
 
# app = Flask(__name__)

 
tokenizer = FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)
 
init_checkpoint = FLAGS.init_checkpoint
use_tpu=False
 
sess=tf.Session()
 
bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
 
print(init_checkpoint)
 
is_training=False
use_one_hot_embeddings=False
 
def convert_single_example(vectors, maxlen=10):
    length=len(vectors)
    if length>=maxlen:
        return  vectors[0:maxlen], [1]*maxlen, [0]*maxlen
    else:
        input=vectors+[0]*(maxlen-length)
        mask=[1]*length+[0]*(maxlen-length)
        segment=[0]*maxlen
        return input, mask, segment
 
 
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
 
 
 
# @app.route('/bertvectors')
# def response_request():
#     text = request.args.get('text')
 
#     vectors = [di.get("[CLS]")] + [di.get(i) if i in di else di.get("[UNK]") for i in list(text)] + [di.get("[SEP]")]
 
#     input, mask, segment = inputs(vectors)
 
#     input_ids = np.reshape(np.array(input), [1, -1])
#     input_mask = np.reshape(np.array(mask), [1, -1])
#     segment_ids = np.reshape(np.array(segment), [1, -1])
 
#     embedding = tf.squeeze(model.get_sequence_output())
 
#     ret=sess.run(embedding,feed_dict={"input_ids_p:0":input_ids,"input_mask_p:0":input_mask,"segment_ids_p:0":segment_ids})
#     return  json.dumps(ret.tolist(), ensure_ascii=False)

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

def response_request(text_list):
  """ return the embedding of the request text list
      Text list contains list of Chinese words.
      Pool the tokens vectors to become word representation
  """
  word2bert = {}
  for text in text_list:
    indice = [tokenizer.vocab.get("[CLS]")] \
      + [tokenizer.vocab.get(token) for token in tokenizer.tokenize(text)] \
      + [tokenizer.vocab.get("[SEP]")]
    input, mask, segment = convert_single_example(indice, maxlen=64)
    input_ids   = np.reshape(np.array(input), [1, -1])
    input_mask  = np.reshape(np.array(mask), [1, -1])
    segment_ids = np.reshape(np.array(segment), [1, -1])
    embeddings   = tf.squeeze(model.get_sequence_output())
    pool_embedding = pool_vectors(embeddings, tf.cast(input_mask, tf.float32))
    ret = sess.run(pool_embedding, 
        feed_dict={"input_ids_p:0":input_ids, "input_mask_p:0":input_mask, 
            "segment_ids_p:0":segment_ids})
    word2bert[text] = ret.tolist()

  return word2bert
 
if __name__ == "__main__":
    # server = pywsgi.WSGIServer(('0.0.0.0', 19877), app)
    # server.serve_forever()
    pass