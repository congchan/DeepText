## Deep Learning for Text
Deep learning could help to process text for various purposes, say, topic/sentiment classification, tagging, text summarization, keywords extraction, and text clustering etc.. These NLP tasks cover a large fraction of NLP application domains.  

Most of these tasks are essentially classification problem or clustering problem, which could be handled with (multinomial) logistic regression model, and sequence-to-sequence(seq2seq) architecture.

This package aims to handle most of these tasks using deep learning models with a unitfied and cleaned API.

## Requirement
`Python > 3.6`
`Keras > 2.0`

## Models List
* fasttext
* textRNN
* textCNN
* textRCNN
* textInception
* bigru_att

## Features
1. Get BERT pre-trained words representation. You should download the BERT pre-trained model by yourself, and put it at the `./bert/` directory, say `./bert/chinese_L-12_H-768_A-12`
2. Data: provide data / process data
3. Define task: say, a classifier. Make use of `classifier.py`

### Data
Make use of `SampleProcessor` class (or define your own) 
```python
# process data from files 
>>> processer = SampleProcessor(config, )

# provide your own list of data = [train_X, train_Y, test_X, test_Y]
>>> processer = SampleProcessor(config, data)
```

### Task
For classifier, make use of `Classifier` class
```python
# init a new classifier:
>>> myclassifier = Classifier(max_seq_length=500, emb_size=300)

init a new classifier with config parameters (config as dictionary):
>>> myclassifier = Classifier(**config)

load from a check point: 
>>> myclassifier = Classifier(check_point = './your_own_model.h5')
```

### BERT pre-trained words representation
support pre-trained bert model: chinese_L-12_H-768_A-12
```python
# 提取中文 bert embedding 用于你的模型， 
# Notice: pre-trained bert model embedding size should be the same as your model
pre_train_emb = processer.load_bert_embedding(processer.vob_size, 
        processer.word2id)
config.update({"pre_train_emb": pre_train_emb})
```