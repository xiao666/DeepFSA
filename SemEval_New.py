"""Trains the DeepMoji architecture on the SemEval2017 Task 5.
   New method only uses the model architecture without pretraining.
"""
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
from sklearn.metrics.pairwise import cosine_similarity
from examples.custom_metric import cos
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping


#load SemEval-2017
#load microblogs
train1=pd.read_csv('Microblog_Trainingdata1.csv')
test1 = pd.read_csv('Microblogs_Testdata_withscores.csv')
X_train1=list(train1['spans'])
y_train1=list(train1['sentiment score'])
X_test1=list(test1['spans'])
y_test1=list(test1['sentiment score'])


#load headlines
train2=pd.read_csv('Headlines_Trainingdata.csv')
test2 = pd.read_csv('Headlines_Testdata_withscores.csv')
X_train2=list(train2['title'])
y_train2=list(train2['sentiment score'])
X_test2=list(test2['title'])
y_test2=list(test2['sentiment score'])



#select data / data++
X_train=X_train2#+X_train2
y_train=y_train2#+y_train2


#select test dataset
X_test=X_test2
y_test=y_test2





print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')


X=X_train+X_test
labels=y_train+y_test
texts=[]
for i in range(len(X)):
    texts.append(str(X[i]))

batch_size, maxlen = calculate_batchsize_maxlen(texts)

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

st = SentenceTokenizer(vocab, maxlen)

nb_classes=0

for i in range(len(X_train)):
    X_train[i]=str(X_train[i])

for i in range(len(X_test)):
    X_test[i]=str(X_test[i])
X_train=[st.tokenize_sentences(s)[0] for s in [X_train]]
X_test=[st.tokenize_sentences(s)[0] for s in [X_test]]

y_train=np.array(y_train)
y_test=np.array(y_test)

print ("batch size, maxlen: ", batch_size,maxlen)


###model and experiments

model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=maxlen,
                              embed_dropout_rate=0.3,final_dropout_rate=0.2,embed_l2=1E-6)


lr = 0.001# lr for new , last
adam = Adam(clipnorm=1, lr=lr)


model.summary()
model.compile(loss='mse',optimizer='adam', metrics=[cos])

earlystop = EarlyStopping(monitor='val_loss', patience=5)
#savebest=ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path,save_best_only=True)
callbacks=[earlystop]#   ,savebest

print('Train...')
history = model.fit(X_train,y_train, batch_size=batch_size, epochs=14 ,validation_data=(X_test, y_test)) #,callbacks=callbacks

'''
mae = model.evaluate(X_test,y_test, batch_size=batch_size)

print('Test mae:', mae)
predicted_sentiment=model.predict(X_test)

cosine=sem_eval(y_test,predicted_sentiment)
y_test=np.array(y_test).reshape(1,-1)
predicted_sentiment=np.array(predicted_sentiment).reshape(1,-1)

cosine2=cosine_similarity(y_test,predicted_sentiment)
print ("cosine similarity:", cosine)
print ("cosine similarity2:", cosine2)
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
