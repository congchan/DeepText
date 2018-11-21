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

## Usage
step:
1. Data: provide data / process data
2. define task: say, a classifier

Make use of `classifier.py`

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
>>> myclassifier = Classifier(max_seq_length=500, emb=300)

init a new classifier with config parameters:
>>> myclassifier = Classifier(**config)

load from a check point: 
>>> myclassifier = Classifier(check_point = './your_own_model.h5')
```