
import pandas as pd
import numpy as np
import csv
from keras.layers import Input, Dense, LSTM, merge, Dropout, Activation
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

#import data
#numerical = np.asarray(pd.read_csv('DJIA_table.csv')['Close'])
#numerical = list(pd.read_csv('returns.csv')['returns'])
numerical = list(pd.read_csv('ndr.csv')['ndr'])
print (numerical[0])
print (max(numerical),min(numerical))
data = pd.read_csv('scores2.csv')['scores']
data=np.asarray(data).reshape(1989,25)
print (data[0])
temp=[]

#for i in range(len(X)):
#    temp.append(sum(X[i]))


#print (temp[0])
'''
#concatenate
temp=[[0 for x in range(26)] for y in range(1989)] 
for y in range(1989):
    temp[y][25]=numerical[y]
    for x in range(25):
        temp[y][x]=data[y][x]

print (temp[0])
'''


#X_train=np.array(temp[0:1611]).reshape(-1,1,26)
#X_test=np.array(temp[1611:1989]).reshape(-1,1,26)
X_train=np.array(numerical[0:1611]).reshape(-1,1,1)
X_test=np.array(numerical[1611:1989]).reshape(-1,1,1)
#print (min(X))
#print (X_train[0])
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
'''
model = Sequential()
model.add(Dense(128, input_dim=25))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=256, epochs=5,
          validation_data=(X_test, Y_test))

'''



model=Sequential()

model.add(LSTM(64,input_shape=(1,1)))
#model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_acc', patience=5)
history=model.fit(X_train, Y_train, epochs=100,
          validation_data=(X_test, Y_test),callbacks=[early_stopping])
          
          
print (history.history)