from __future__ import print_function
import example_helper
import numpy as np
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import  HashingVectorizer
from examples.model_def import deepmoji_architecture, deepmoji_transfer, load_specific_weights                #change all dir. from deepmoji ti examples!!

import csv
import pandas as pd
from examples.finetuning import calculate_batchsize_maxlen, freeze_layers
from examples.global_variables import NB_TOKENS, VOCAB_PATH, PRETRAINED_PATH
from examples.sentence_tokenizer import SentenceTokenizer
import json
from examples.sem_evaluate import sem_eval
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from examples.custom_metric import cos
import matplotlib.pyplot as plt
from keras.utils import np_utils

def column(matrix, i):
    return [row[i] for row in matrix]
X = np.array(pd.read_csv('merged_daily_news.csv')['news'])
data = pd.read_csv('daily_news.csv')
X=np.array(list(data['news'])).reshape(-1,25)
X=column(X,0)
X=X[0:300]
X_train=X[0:200]
X_test=X[200:300]

texts=[]
for i in range(len(X)):
    texts.append(str(X[i]))

batch_size, maxlen = calculate_batchsize_maxlen(texts)

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

st = SentenceTokenizer(vocab, maxlen)

nb_classes=2

for i in range(len(X_train)):
    X_train[i]=str(X_train[i])

for i in range(len(X_test)):
    X_test[i]=str(X_test[i])
X_train=[st.tokenize_sentences(s)[0] for s in [X_train]]
X_test=[st.tokenize_sentences(s)[0] for s in [X_test]]

Y=np.array(list(pd.read_csv('Combined_News_DJIA.csv')['Label'])[0:300]).reshape(-1,1)
y_train=Y[0:200]
y_test=Y[200:300]
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

print ("batch size, maxlen: ", batch_size,maxlen)


#model
###model and experiments

model = deepmoji_architecture(nb_classes=2, nb_tokens=NB_TOKENS, maxlen=maxlen)

weight_path=PRETRAINED_PATH
#load_specific_weights(model, weight_path, exclude_names=['softmax'])#
#model = freeze_layers(model, unfrozen_keyword='softmax')#last
lr = 0.001
#adam = Adam(clipnorm=1, lr=lr)


#earlystop = EarlyStopping(monitor='val_loss', patience=2)

model.summary()
model.compile(loss='mse',
              optimizer='adam'
             , metrics=['acc'])

print('Train...')
history=model.fit(X_train,y_train, batch_size=batch_size, epochs=2
          ,validation_data=(X_test, y_test))#,callbacks=[earlystop]

#save the weights of full model
#model.save_weights('full_headline_ep8.hdf5')



#met = model.evaluate(X_test,y_test, batch_size=batch_size)

#print('Test mae:', met)
#predicted_sentiment=model.predict(X_test)

'''
print (history.history)
plt.plot()
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss'])
plt.plot(history.history['cos']) 
plt.plot(history.history['val_cos']) 

plt.title('model validation')  
plt.ylabel('avg.loss(avg.cos)')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
plt.show() 
'''