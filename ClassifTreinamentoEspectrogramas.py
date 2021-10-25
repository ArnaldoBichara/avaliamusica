###################
# Este componente executa o treino dos modelos de machine learning para recomendação por conteúdo
# com dados de entrada na forma de espectrogramas das músicas
# A entrada consiste nos espectrogramas de amostras de audio do Usuário A com a classe (0-não curte, 1-curte)
# Uso de validação cruzada (cross-validation) para cálculo das métricas.
# O modelo com melhor acurácia é salvo como modeloClassifEspectro.pickle.
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
import statistics
from utils import calcula_cross_val_scores


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from statistics import mean
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics._classification import accuracy_score
from tensorflow.python.keras.models import load_model
tf.get_logger().setLevel('ERROR') 


logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifTreinamentoEspectrograma')

#lendo datasets
npzTreino = np.load('./FeatureStore/AudioEspectrogramasTreinoBin.npz')
X_trein= npzTreino['arr_0']
y_trein = npzTreino['arr_1']

#%% dividindo treinamento em treino e validação
X_trein, X_val, y_trein, y_val = train_test_split(X_trein, y_trein, random_state=0, test_size=0.25)

npzTeste = np.load('./FeatureStore/AudioEspectrogramasTesteBin.npz')
X_teste= npzTeste['arr_0']
y_teste = npzTeste['arr_1']

#expande dimensões para uso por conv2d
X_trein = np.expand_dims(X_trein, axis = -1)
X_val = np.expand_dims(X_val, axis = -1)
X_teste = np.expand_dims(X_teste, axis = -1)

def build_modelo_CNN():
    #blocos convolucionais
    n_frequency = 640
    n_frames = 128    
    model = Sequential()
    model.add (Conv2D(16, kernel_size = (3,3), strides=1, padding= 'same', activation='relu', input_shape=(n_frequency, n_frames,1)))
    model.add(MaxPooling2D((2,2)))
    model.add (Dropout(0.05))

    model.add (Conv2D(32, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add (Dropout(0.05))

    model.add (Conv2D(48, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add (Dropout(0.05))

    model.add (Conv2D(64, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
    model.add (MaxPooling2D((4,4)))
    model.add (Dropout(0.05))

    model.add (Conv2D(64, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
    model.add (MaxPooling2D((4,4)))
    model.add (Dropout(0.05))

    model.add (Flatten())

    ## MLP
    model.add (Dense(256,  activation='relu',
                kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add (Dropout(0.1))
    model.add (Dense(1, activation='sigmoid', 
                kernel_initializer='uniform'))
  
    opt = Adam(lr=0.001)
    model.compile(
            loss='binary_crossentropy',
            optimizer=opt, 
            metrics=['accuracy']
        )
    print(model.summary())    
    return model

#%% definindo o modelo, com os hiperparametros previamente escolhidos
cnn = build_modelo_CNN()
checkpoint = ModelCheckpoint('./FeatureStore/melhorModeloCNN', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=20)
reducelr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_delta=0.01,verbose=1)
callbacks_list = [reducelr, checkpoint, early_stop]

history = cnn.fit(X_trein, y_trein, batch_size=20, epochs=100,validation_data=(X_val, y_val), 
                    verbose=2, callbacks=callbacks_list)

# Cálculo de acurácia:
#
# y_predicao = cnn.predict (X_teste)
# y_predicao = [round(y[0]) for y in y_predicao]
# acuracia = accuracy_score (y_teste, y_predicao)
# print("acuracia CNN {:.3f}".format(acuracia))

cnn = load_model ('./FeatureStore/melhorModeloCNN')
y_predicao = cnn.predict (X_teste)
y_predicao = [round(y[0]) for y in y_predicao]
acuracia = accuracy_score (y_teste, y_predicao)
print("acuracia CNN {:.3f}".format(acuracia))

logging.info ("acuracia CNN {:.3f}".format(acuracia))

logging.info('\n<< ClassifTreinamentoEspectrograma')
