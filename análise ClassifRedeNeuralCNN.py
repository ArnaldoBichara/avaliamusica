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

#%%
logging.basicConfig(filename='./Analises/processamClassifCNN.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
#%%
dict_classes = {'Não Curte':0, 'Curte': 1  }


reverse_map = {v: k for k, v in dict_classes.items()}
#print(reverse_map)
#%%
# X - espectrogramas
# y - classe
npzfile = np.load('./FeatureStore/AudioEspectrogramasTreinoBin.npz')
X_train = npzfile['arr_0']
y_train = npzfile['arr_1']


#%% inicialmente vamos dividir em treino, validação e teste
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.25)

# %%
num_classes = 2 # classes: não curte, curte
n_features = X_train.shape[1] # n_freq
n_time = X_train.shape[2]     # n_frames

lstm_count = 64
num_units = 120

#%%
def build_modelo_convolucional():
    #print('Building modelo...')

    #blocos convolucionais
    n_frequency = 640
    n_frames = 128    
    model = Sequential()
    model.add (Conv2D(16, kernel_size = (3,2), strides=1, padding= 'same', activation='relu', input_shape=(n_frequency, n_frames,1)))
    #model.add(BatchNormalization())
    #model.add (Conv2D(16,kernel_size =(5,1),strides=2,padding='same',activation='relu')) # faz papel de model.add(MaxPooling2D((2,2)))
    model.add(MaxPooling2D((2,2)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(32, kernel_size=(3,2), strides=1, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    #model.add (Conv2D(32,kernel_size =(5,1),strides=2,padding='same',activation='relu')) # faz papel de model.add(MaxPooling2D((2,2)))
    model.add(MaxPooling2D((2,2)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(32, kernel_size=(3,2), strides=1, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    #model.add (Conv2D(32,kernel_size =(5,1),strides=2,padding='same',activation='relu')) # faz papel de model.add(MaxPooling2D((2,2)))
    model.add(MaxPooling2D((2,2)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(32, kernel_size=(3,2), strides=1, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    model.add (MaxPooling2D((4,4)))
    #model.add(BatchNormalization())
    model.add (Dropout(0.05))

    model.add (Conv2D(64, kernel_size=(3,2), strides=1, padding='same', activation='relu'))
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
                kernel_initializer='uniform', 
                kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add (Dropout(0.3))
    # model.add (Dense(75,  name='dense2', activation='relu',
    #             kernel_initializer='uniform', 
    #             kernel_regularizer=l2(0.001)))
    # model.add (Dropout(0.1))
    model.add (Dense(1, activation='sigmoid', 
                kernel_initializer='uniform', 
                kernel_regularizer=l2(0.001)))
  
    opt = Adam(lr=0.001)
#    opt= RMSprop(lr=0.001)
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
    
    checkpoint_callback = ModelCheckpoint('./FeatureStore/weights.best.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')
    reducelr_callback = ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=7, min_delta=0.01,
                verbose=1
            )
    callbacks_list = [checkpoint_callback, reducelr_callback]

    # Fit the model and get training history.
    print('Executando Treinamento...')
    history = model.fit(x_train, y_train, batch_size=20, epochs=50,
                        validation_data=(x_val, y_val), verbose=2, callbacks=callbacks_list)

    return model, history

def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#%%
model, history  = treina_modelo(X_train, y_train, X_valid, y_valid)
