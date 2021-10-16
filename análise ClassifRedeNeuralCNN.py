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
from os.path import isfile
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

#%%
logging.basicConfig(filename='./Analises/processamClassifCNN.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
#%%
# X - espectrogramas
# y - classe
npzfile = np.load('./FeatureStore/AudioEspectrogramas.npz')
X = npzfile['arr_0']
y = npzfile['arr_1']


#%% vamos ver um dos espectrogramas
espectrograma = X[30]
classe = y[30]
print ("Classe: ", ("Não Curte" if classe == 0 else "Curte"))
plt.figure(figsize=(10, 5))
librosa.display.specshow(espectrograma.T, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Teste Melspectogram')
plt.tight_layout()

#%% inicialmente vamos dividir em treino, validação e teste
# !!! depois acertar isso !!!!
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.25)

# %%
num_classes = 2
n_features = X_train.shape[2]
n_time = X_train.shape[1]

nb_filters1=16 
nb_filters2=32 
nb_filters3=64
nb_filters4=64
nb_filters5=64
kernel_size = (3,1)
pool_size_1= (2,2) 
pool_size_2= (4,4)
pool_size_3 = (4,2)

dropout_prob = 0.20
dense_size1 = 128
lstm_count = 64
num_units = 120

BATCH_SIZE = 64
EPOCH_COUNT = 50
L2_regularization = 0.001
#%%
def build_modelo_convolucional(model_input):
    print('Building modelo...')
    layer = model_input

    #blocos convolucionais
    conv_1 = Conv2D(name='conv_1', filters = nb_filters1, kernel_size = kernel_size, strides=1,
                      padding= 'valid', activation='relu')(layer)
    pool_1 = MaxPooling2D(pool_size_1)(conv_1)

    conv_2 = Conv2D(name='conv_2', filters = nb_filters2, kernel_size = kernel_size, strides=1,
                      padding= 'valid', activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size_1)(conv_2)

    conv_3 = Conv2D(name='conv_3', filters = nb_filters3, kernel_size = kernel_size, strides=1,
                      padding= 'valid', activation='relu')(pool_2)
    pool_3 = MaxPooling2D(pool_size_1)(conv_3)
        
    conv_4 = Conv2D(name='conv_4', filters = nb_filters4, kernel_size = kernel_size, strides=1,
                      padding= 'valid', activation='relu')(pool_3)
    pool_4 = MaxPooling2D(pool_size_2)(conv_4)
    
    conv_5 = Conv2D(name='conv_5', filters = nb_filters5, kernel_size = kernel_size, strides=1,
                      padding= 'valid', activation='relu')(pool_4)
    pool_5 = MaxPooling2D(pool_size_2)(conv_5)
    
    flatten1 = Flatten()(pool_5)

    ## Output
    model_output = Dense(units=num_classes,  name='preds', activation = 'sigmoid')(flatten1)
    
    model = Model(model_input, model_output)
    
#     opt = Adam(lr=0.001)
    opt = RMSprop(lr=0.0005)  # Optimizer
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
    return model

def treina_modelo(x_train, y_train, x_val, y_val):
    
    n_frequency = 128
    n_frames = 640
    
    #expande dimensões para uso por conv2d
    x_train = np.expand_dims(x_train, axis = -1)
    x_val = np.expand_dims(x_val, axis = -1)
    
    input_shape = (n_frames, n_frequency, 1)
    model_input = Input(input_shape, name='input')
    
    model = build_modelo_convolucional(model_input)
    
    checkpoint_callback = ModelCheckpoint('./models/parallel/weights.best.h5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')
    reducelr_callback = ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            )
    callbacks_list = [checkpoint_callback, reducelr_callback]

    # Fit the model and get training history.
    print('Executando Treinamento...')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)

    return model, history

def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
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
# %%
