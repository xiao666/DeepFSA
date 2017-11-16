"""Trains the DeepMoji architecture on the SemEval2017 Task 5.
   Chain-thaw method load pretraining weights and updates layers parameters using "chain-thaw" method.
"""
from __future__ import print_function
import example_helper
import numpy as np
from keras.preprocessing import sequence

from examples.model_def import deepmoji_architecture, deepmoji_transfer, load_specific_weights                #change all dir. from deepmoji to examples!!

import csv
import pandas as pd
from examples.finetuning import calculate_batchsize_maxlen, freeze_layers, train_by_chain_thaw, finetuning_callbacks, sampling_generator
from examples.global_variables import NB_TOKENS, VOCAB_PATH, PRETRAINED_PATH, WEIGHTS_DIR
from examples.sentence_tokenizer import SentenceTokenizer
import json
from examples.sem_evaluate import sem_eval
from keras.optimizers import Adam
import uuid
from time import sleep
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


lr = 0.001# lr for new , last
#lr=0.0001#lr for chain-thaw, full
adam = Adam(clipnorm=1, lr=lr)

############Here use the chain-thaw, finetune parameters layer by layer; 1st last layer; then from 1st layer;
#save weights after covengce of each layer.

earlystop = EarlyStopping(monitor='val_loss', patience=1)
savebest=ModelCheckpoint(monitor='val_loss', filepath='chain_final.hdf5',save_best_only=True)
callbacks=[earlystop, savebest ]#  ,savebest

'''
#last layer: name=softmax (actually is a regression tanh unit)
model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=maxlen)

weight_path=PRETRAINED_PATH
load_specific_weights(model, weight_path, exclude_names=['softmax'])
model = freeze_layers(model, unfrozen_keyword='softmax')
#model.summary()

model.compile(loss='mean_squared_error',optimizer='adam', metrics=[cos])#

print('Finetune last layer')
history=model.fit(X_train,y_train, batch_size=batch_size, epochs=58,validation_data=(X_test, y_test))
model.save_weights('chain_last.hdf5')

predicted_sentiment=model.predict(X_test)
cosine=sem_eval(y_test,predicted_sentiment)

print ("cosine similarity:", cosine)
print (history.history)
'''


'''
#embedding layer
print('Finetune Embedding')
#del model
sleep(1)
model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=maxlen,embed_dropout_rate=0.3,final_dropout_rate=0.2,embed_l2=1E-6)
model.load_weights('weight_last.hdf5')
sleep(1)
model = freeze_layers(model,unfrozen_keyword='embedding')
#model.summary()
model.compile(loss='mean_squared_error',optimizer='adam', metrics=[cos])#

history=model.fit(X_train,y_train, batch_size=batch_size, epochs=50,validation_data=(X_test, y_test),callbacks=callbacks)
#model.save_weights('weight_embedding.hdf5')
#mae = model.evaluate(X_test,y_test, batch_size=batch_size)

print (history.history)

'''

'''
#bi_lstm_0
print('Finetune bi_lstm_0')
#del model
sleep(1)
model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=maxlen,embed_dropout_rate=0.3,final_dropout_rate=0.2,embed_l2=1E-6)
model.load_weights('chain_embed.hdf5')
sleep(1)
model = freeze_layers(model,unfrozen_keyword='bi_lstm_0')
#model.summary()
model.compile(loss='mean_squared_error',optimizer='adam', metrics=[cos])#

history=model.fit(X_train,y_train, batch_size=batch_size, epochs=50,validation_data=(X_test, y_test),callbacks=callbacks)

print (history.history)
'''


'''
#bi_lstm_1
print('Finetune bi_lstm_1')
#del model
sleep(1)
model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=maxlen,embed_dropout_rate=0.3,final_dropout_rate=0.2,embed_l2=1E-6)
model.load_weights('chain_bilstm0.hdf5')                 #since bi_lstm_0 overfit at the very beginning
sleep(1)
model = freeze_layers(model,unfrozen_keyword='bi_lstm_1')
#model.summary()
model.compile(loss='mean_squared_error',optimizer='adam', metrics=[cos])#

history=model.fit(X_train,y_train, batch_size=batch_size, epochs=50,validation_data=(X_test, y_test),callbacks=callbacks)

print (history.history)
'''

'''
#attlayer
print('Finetune attlayer')
#del model
sleep(1)
model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=maxlen,embed_dropout_rate=0.3,final_dropout_rate=0.2,embed_l2=1E-6)
model.load_weights('chain_bilstm1.hdf5')                 #since bi_lstm_0 overfit at the very beginning

sleep(1)
model = freeze_layers(model,unfrozen_keyword='attlayer')
#model.summary()
model.compile(loss='mean_squared_error',optimizer='adam', metrics=[cos])#
history=model.fit(X_train,y_train, batch_size=batch_size, epochs=50,validation_data=(X_test, y_test),callbacks=callbacks)

print (history.history)
'''



#all layer
print('Finetune all layer')
lr=0.0001#lr for chain-thaw, full
adam = Adam(clipnorm=1, lr=lr)

#del model
sleep(1)
model = deepmoji_architecture(nb_classes=0, nb_tokens=NB_TOKENS, maxlen=maxlen,embed_dropout_rate=0.3,final_dropout_rate=0.2,embed_l2=1E-6)
model.load_weights('chain_att.hdf5')
sleep(1)

model.compile(loss='mean_squared_error',optimizer=adam, metrics=[cos])#

history=model.fit(X_train,y_train, batch_size=batch_size, epochs=50,validation_data=(X_test, y_test),callbacks=callbacks)

predicted_sentiment=model.predict(X_test)
cosine=sem_eval(y_test,predicted_sentiment)

print ("cosine similarity:", cosine)
print (history.history)



























'''
#checkpoint_path: Where weight checkpoints should be saved.
#patience: Number of epochs with no improvement after which training will be stopped.
checkpoint_weight_path='{}/deepmoji-checkpoint-{}.hdf5' \
                      .format(WEIGHTS_DIR, str(uuid.uuid4()))
callbacks = finetuning_callbacks(checkpoint_weight_path, patience=2, verbose=2)
train_gen = sampling_generator(X_train, y_train, batch_size,epoch_size=1700,
                                   upsample=False, seed=42)#==========================================epoch_size=1700

train_by_chain_thaw(model,X_train,y_train,val_data=(X_test,y_test),loss='mse',  epoch_size=1700 , nb_epochs=3,checkpoint_weight_path=checkpoint_weight_path,batch_size=batch_size )

mae = model.evaluate(X_test,y_test, batch_size=batch_size)
predicted_sentiment=model.predict(X_test)
'''


'''
model.compile(loss='mean_squared_error', optimizer='adam' , metrics=['mae'])

print('Train...')
model.fit(X_train,y_train, batch_size=batch_size, epochs=5
          ,validation_data=(X_test, y_test))
mae = model.evaluate(X_test,y_test, batch_size=batch_size)

print('Test mae:', mae)
predicted_sentiment=model.predict(X_test)
'''

'''
cosine=sem_eval(y_test,predicted_sentiment)
print ("cosine similarity:", cosine)
'''



