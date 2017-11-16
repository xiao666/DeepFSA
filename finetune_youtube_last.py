"""Finetuning example.

Trains the DeepMoji model on the SS-Youtube dataset, using the 'last'
finetuning method and the accuracy metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.
"""

from __future__ import print_function
import example_helper
import json
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH
from deepmoji.finetuning import (
     load_benchmark,
     finetune)

import pandas as pd
import numpy as np
'''
DATASET_PATH = '../data/SS-Youtube/raw.pickle'
nb_classes = 2

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

# Load dataset.
data = load_benchmark(DATASET_PATH, vocab)

# Set up model and finetune
model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH)
model.summary()
model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                      data['batch_size'], method='last')
print('Acc: {}'.format(acc))
'''

#load SemEval-2017
print ('000000000000000000000000')


train=pd.read_csv('Microblog_Trainingdata.csv')
test = pd.read_csv('Microblogs_Testdata_withscores.csv')
X_train=np.asarray(train['spans__001'])
y_train=np.asarray(train['sentiment score'])

X_test=np.asarray(test['spans'])
y_test=np.asarray(test['sentiment score'])
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)



nb_classes=0
# Set up model and finetune
model = deepmoji_transfer(0, 50, PRETRAINED_PATH)
model.summary()
model, acc = finetune(model, X_train, y_train, nb_classes,
                      32, method='last')
print('Acc: {}'.format(acc))
