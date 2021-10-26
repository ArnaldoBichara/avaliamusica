###################
# Análise de classificação por conteúdo,
# usando Rede Neural Convolucional e
# espectrogramas
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

import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.regularizers import l2
from keras.constraints import maxnorm
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping

#%%
logging.basicConfig(filename='./Analises/processamClassifCNN.log', 
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

def build_modelo_convolucional():
    #blocos convolucionais
    n_frequency = 640
    n_frames = 128    
    model = Sequential()
    model_input = Input(n_frames, n_frequency, 1, name='input')
    model.add (Conv2D(16, kernel_size = (3,4), strides=1, padding= 'same', activation='relu', input_shape=(n_frequency, n_frames,1)))
    #model.add(BatchNormalization())
    #model.add (Conv2D(16,kernel_size =(5,1),strides=2,padding='same',activation='relu')) # faz papel de model.add(MaxPooling2D((2,2)))
    model.add(MaxPooling2D((2,2)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(32, kernel_size=(3,4), strides=1, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    #model.add (Conv2D(32,kernel_size =(5,1),strides=2,padding='same',activation='relu')) # faz papel de model.add(MaxPooling2D((2,2)))
    model.add(MaxPooling2D((2,2)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(48, kernel_size=(3,4), strides=1, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    #model.add (Conv2D(32,kernel_size =(5,1),strides=2,padding='same',activation='relu')) # faz papel de model.add(MaxPooling2D((2,2)))
    model.add(MaxPooling2D((2,2)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(64, kernel_size=(3,4), strides=1, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    model.add (MaxPooling2D((4,4)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(64, kernel_size=(3,4), strides=1, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    model.add (MaxPooling2D((4,4)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Flatten())

    """     ### Recurrent Block
    
    # Pooling layer
    pool_lstm1 = MaxPooling2D((4,2), name = 'pool_lstm')(model_input)
    print("pool_lstm1 shape: ", pool_lstm1.shape)
   # Embedding layer
    squeezed = Lambda(lambda x: K.squeeze(x, axis= -1))(pool_lstm1)   
    print("squeezed shape: ", squeezed.shape) 
   # Bidirectional GRU
    lstm = Bidirectional(GRU(lstm_count))(squeezed)  #default merge mode is concat    
    print("lstm shape: ", lstm.shape)

    # Concat Output
    concat = concatenate([flatten1, lstm], axis=-1, name ='concat') """

    ## MLP
    model.add (Dense(256,  activation='relu',
                kernel_initializer='uniform'))
                #,kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add (Dropout(0.1))
    #model.add (Dense(75,  name='dense2', activation='relu',
    #             kernel_initializer='uniform')) 
    #             kernel_regularizer=l2(0.001)))
    #model.add(BatchNormalization())
    #model.add (Dropout(0.1))
    model.add (Dense(1, activation='sigmoid', 
                kernel_initializer='uniform'))
#                kernel_regularizer=l2(0.001)))
  
    opt = Adam(lr=0.001)
#    opt= RMSprop(lr=0.0005)
    model.compile(
            loss='binary_crossentropy',
            optimizer=opt, 
            metrics=['accuracy']
        )
    print(model.summary())    
    return model

def treina_modelo(x_train, y_train, x_val, y_val):
    
    #expande dimensões para uso por conv2d
    x_train = np.expand_dims(x_train, axis = -1)
    x_val = np.expand_dims(x_val, axis = -1)
    
    model = build_modelo_convolucional()
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15)
    checkpoint = ModelCheckpoint('./FeatureStore/melhorModeloCNN', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')
    reducelr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_delta=0.01,verbose=1)
    callbacks_list = [checkpoint, reducelr, early_stop]

    # Fit the model and get training history.
    print('Executando Treinamento...')
    history = model.fit(x_train, y_train, batch_size=20, epochs=60,validation_data=(x_val, y_val), 
                        verbose=2, callbacks=callbacks_list)

    return model, history

model, history  = treina_modelo(X_train, y_train, X_valid, y_valid)
