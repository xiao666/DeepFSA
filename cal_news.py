from __future__ import print_function
import example_helper
import numpy as np
from keras.preprocessing import sequence

from examples.model_def import deepmoji_architecture, deepmoji_transfer, load_specific_weights                #change all dir. from deepmoji ti examples!!

import csv
import pandas as pd
from examples.finetuning import calculate_batchsize_maxlen, freeze_layers
from examples.global_variables import NB_TOKENS, VOCAB_PATH, PRETRAINED_PATH
from examples.sentence_tokenizer import SentenceTokenizer
import json
from examples.sem_evaluate import sem_eval
from keras.optimizers import Adam
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics.pairwise import cosine_similarity
from examples.custom_metric import cos
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import HashingVectorizer



data = pd.read_csv('daily_news.csv')
X=list(data['news'])

#tokenize
print (str(X[1]))

texts=[]
for i in range(len(X)):
    texts.append(str(X[i]))

#batch_size, maxlen = calculate_batchsize_maxlen(texts)

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

st = SentenceTokenizer(vocab, 20)

nb_classes=0

for i in range(len(X)):
    X[i]=str(X[i])

X=[st.tokenize_sentences(s)[0] for s in [X]]

print (len(X))
print (len(X[0][0]))
print (X[0][0])
#X=X[0][0:2]
#print (batch_size,maxlen)

#load sentiment model
model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=20,embed_dropout_rate=0.3,final_dropout_rate=0.2,embed_l2=1E-6)
weight_path='news_full.hdf5'
load_specific_weights(model, weight_path)

model.summary()
model.compile(loss='mse',optimizer='adam' , metrics=[cos])

print('Calculating...')

predicted_sentiment=model.predict(X)

predicted_sentiment=np.array(predicted_sentiment).reshape(-1,1)

#write into csv
OUTPUT_PATH='scores2.csv'

print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['scores'])#in all 64 emojis
    for i, row in enumerate(predicted_sentiment):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))



'''
data = pd.read_csv('Combined_News_DJIA.csv')
headlines=[[0 for x in range(25)] for y in range(len(data.index))]#len(data.index)
for row in range(len(data.index)):#len(data.index)
    for col in range(25):
        temp0=str(data.iloc[row,(col+2)])
        #temp0=temp0.lower()
        temp=HashingVectorizer().build_tokenizer()(temp0)
        #=========code below remove stopwords==================
        #temp=[s for s in temp if s not in stopwords]
        #headlines[row][col]=temp
        headlines[row][col]=(' '.join(word for word in temp))


headlines=np.array(headlines).reshape(-1,1)

OUTPUT_PATH='daily_news.csv'

print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['news'])#in all 64 emojis
    for i, row in enumerate(headlines):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))

'''