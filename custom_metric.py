import keras.backend as K

def cos_loss(y_true, y_pred):

    return -K.sum(y_true * y_pred)/(K.sqrt(K.sum(K.square(y_pred)))*K.sqrt(K.sum(K.square(y_true))))



def cos(y_true, y_pred):

    return K.sum(y_true * y_pred)/(K.sqrt(K.sum(K.square(y_pred)))*K.sqrt(K.sum(K.square(y_true))))