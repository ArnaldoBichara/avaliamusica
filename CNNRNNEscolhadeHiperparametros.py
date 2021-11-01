###################
# Análise de classificação por conteúdo,
# usando Rede Neural CNN paralelo com rede LSTM (RNN)
# As entradas são espectrogramas.
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers

# import librosa
# import librosa.display
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.regularizers import l2
from keras.constraints import maxnorm
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping

#%%
logging.basicConfig(filename='./Analises/processamClassifCNNeRNN.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

# X - espectrogramas
# y - classe
npzfile = np.load('./FeatureStore/AudioEspectrogramasTreinoBin.npz')
X_train = npzfile['arr_0']
y_train = npzfile['arr_1']

#%% inicialmente vamos dividir em treino e validação
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.25)

def build_modelo_CNNeRNN():

    n_frequency = 640
    n_frames = 128    
    input_shape = (n_frequency, n_frames, 1)
    model_input = Input(input_shape, name='input')    

    #blocos convolucionais
    conv1 = Conv2D(16, kernel_size = (3,3), strides=1, padding= 'same', activation='relu')(model_input)
    pool1 = MaxPooling2D((2,2))(conv1)
    drop1 = Dropout(0.05)(pool1)
    conv2 = Conv2D(32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(drop1)
    pool2 = MaxPooling2D((2,2))(conv2)
    drop2 = Dropout(0.05)(pool2)
    conv3 = Conv2D(48, kernel_size=(3,3), strides=1, padding='same', activation='relu')(drop2)
    pool3 = MaxPooling2D((2,2))(conv3)
    drop3 =Dropout(0.05)(pool3)
    conv4 = Conv2D(64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(drop3)
    pool4 = MaxPooling2D((4,4))(conv4)
    drop4 = Dropout(0.05)(pool4)
    conv5 = Conv2D(64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(drop4)
    pool5 = MaxPooling2D((4,4))(conv5)
    drop5 = Dropout(0.05)(pool5)
    flatten = Flatten()(drop5)

    #bloco RNN
        # Pooling layer
    pool_lstm1 = (MaxPooling2D((4,2)))(model_input)
    #pool_lstm1 = (MaxPooling2D((2,1)))(model_input)
        # Embedding layer
    squeezed = Lambda(lambda x: K.squeeze(x, axis= -1))(pool_lstm1)
        # Bidirectional GRU
    lstm_count = 64
    lstm = Bidirectional(GRU(lstm_count))(squeezed)  

    # Concat Output
    concat = concatenate([flatten, lstm], axis=-1, name ='concat') 

    ## MLP
    densed1 = Dense(256,  activation='relu', kernel_initializer='uniform')(concat)
    batchd1 = BatchNormalization()(densed1)
    dropd1  = Dropout(0.1)(batchd1)

    densed2 = Dense(128,  activation='relu', kernel_initializer='uniform')(dropd1)
    batchd2 = BatchNormalization()(densed2)
    dropd2  = Dropout(0.1)(batchd2)

    densed3 = Dense(64,  activation='relu', kernel_initializer='uniform')(dropd2)
    batchd3 = BatchNormalization()(densed3)
    dropd3  = Dropout(0.1)(batchd3)

    model_output = Dense(1, activation='sigmoid', kernel_initializer='uniform')(dropd3)
  
    model = Model(model_input, model_output)

    opt = Adam(lr=0.001)
#    opt= RMSprop(lr=0.0005)
    model.compile(loss='binary_crossentropy',
                    optimizer=opt, 
                    metrics=['accuracy'])
    print(model.summary())    
    return model

def treina_modelo(x_train, y_train, x_val, y_val):
    
    #expande dimensões para uso por conv2d
    x_train = np.expand_dims(x_train, axis = -1)
    x_val = np.expand_dims(x_val, axis = -1)
    
    model = build_modelo_CNNeRNN()
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15)
    checkpoint = ModelCheckpoint('./FeatureStore/modeloClassifCNNeRNN', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')
    reducelr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_delta=0.01,verbose=1)
    callbacks_list = [checkpoint, reducelr, early_stop]
    #callbacks_list = [reducelr, early_stop]

    # Fit the model and get training history.
    print('Executando Treinamento...')
    history = model.fit(x_train, y_train, batch_size=20, epochs=60,validation_data=(x_val, y_val), 
                        verbose=2, callbacks=callbacks_list)

    return model, history

model, history  = treina_modelo(X_train, y_train, X_valid, y_valid)

