
import pandas as pd
import numpy as np
import csv
from keras.layers import Input, Dense, LSTM, merge, Dropout, Activation
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


#import data
X = np.asarray(pd.read_csv('DJIA_table.csv')['Close'])[::-1]
#X = np.asarray(pd.read_csv('returns.csv')['returns'])[::-1]
#X = np.asarray(pd.read_csv('ndr.csv')['ndr'])[::-1]



X_train=np.array(X[0:1611]).reshape(-1,1,1)
X_test=np.array(X[1611:1989]).reshape(-1,1,1)

print (len(X_train))

label=np.array(pd.read_csv('Combined_News_DJIA.csv')['Label']).reshape(-1,1)
Y_train=label[0:1611]
Y_test=label[1611:1989]

nb_classes=2
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)




#build lstm
timesteps=1
data_dim=25

model=Sequential()

model.add(LSTM(64,input_shape=(1,1),dropout=0.5, recurrent_dropout=0.2))
#model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_acc', patience=5)

model.fit(X_train, Y_train, epochs=10,
          validation_data=(X_test, Y_test),callbacks=[early_stopping])
          
          
#0.5,0.2 -->> acc 0.5132
