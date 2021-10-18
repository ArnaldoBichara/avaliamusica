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

#%%
logging.basicConfig(filename='./Analises/processamClassifCNN.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
#%%
dict_classes = {'Não Curte':0, 'Curte': 1  }


reverse_map = {v: k for k, v in dict_classes.items()}
print(reverse_map)
#%%
# X - espectrogramas
# y - classe
npzfile = np.load('./FeatureStore/AudioEspectrogramas.npz')
X = npzfile['arr_0']
y = npzfile['arr_1']

#%% vamos ver um dos espectrogramas
""" espectrograma = X[30]
classe = np.argmax(y[30])
print (reverse_map[classe])
plt.figure(figsize=(10, 5))
librosa.display.specshow(espectrograma.T, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Teste Melspectogram')
plt.tight_layout()
 """
#%% inicialmente vamos dividir em treino, validação e teste
# !!! depois acertar isso !!!!
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.25)

# %%
num_classes = 2 # classes: não curte, curte
n_features = X_train.shape[1] # n_freq
n_time = X_train.shape[2]     # n_frames

n_filtros1=16 # extrai 16 features em paralelo
n_filtros2=32 
n_filtros3=64
n_filtros4=64
n_filtros5=64
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

    print ('model_input shape: ', model_input.shape )
    #blocos convolucionais
    conv_1 = Conv2D(name='conv_1', filters = 16, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu')(model_input)
    print("conv_1 shape: ", conv_1.shape)
    pool_1 = MaxPooling2D((2,2))(conv_1)
    print("pool_1 shape: ", pool_1.shape)

    conv_2 = Conv2D(name='conv_2', filters = 32, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu')(pool_1)
    print("conv_2 shape: ", conv_2.shape)
    pool_2 = MaxPooling2D((2,2))(conv_2)
    print("pool_2 shape: ", pool_2.shape)

    conv_3 = Conv2D(name='conv_3', filters = 64, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu')(pool_2)
    print("conv_3 shape: ", conv_3.shape)
    pool_3 = MaxPooling2D((2,2))(conv_3)
    print("pool_3 shape: ", pool_3.shape)
        
    conv_4 = Conv2D(name='conv_4', filters = 64, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu')(pool_3)
    print("conv_4 shape: ", conv_4.shape)
    pool_4 = MaxPooling2D((4,4))(conv_4)
    print("pool_4 shape: ", pool_4.shape)
    
    conv_5 = Conv2D(name='conv_5', filters = 64, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu')(pool_4)
    print("conv_5 shape: ", conv_5.shape)
    pool_5 = MaxPooling2D((4,4))(conv_5)
    print("pool_5 shape: ", pool_5.shape)

    flatten1 = Flatten()(pool_5)
    print("flatten1 shape: ",flatten1.shape)

    ### Recurrent Block
    
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
    concat = concatenate([flatten1, lstm], axis=-1, name ='concat')

    ## Output
    model_output = Dense(num_classes,  name='preds', activation = 'softmax')(flatten1)
    
    model = Model(model_input, model_output)
    
#     opt = Adam(lr=0.001)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(lr=0.0005), 
            metrics=['accuracy']
        )
    print(model.summary())    
    return model

def treina_modelo(x_train, y_train, x_val, y_val):
    
    #expande dimensões para uso por conv2d
    x_train = np.expand_dims(x_train, axis = -1)
    x_val = np.expand_dims(x_val, axis = -1)

    n_frequency = 640
    n_frames = 128    
    input_shape = (n_frequency, n_frames,  1)
    model_input = Input(input_shape, name='input')
    
    model = build_modelo_convolucional(model_input)
    
    checkpoint_callback = ModelCheckpoint('./FeatureStore/weights.best.h5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')
    reducelr_callback = ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            )
    callbacks_list = [checkpoint_callback, reducelr_callback]

    # Fit the model and get training history.
    print('Executando Treinamento...')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                        validation_data=(x_val, y_val), verbose=2, callbacks=callbacks_list)

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
