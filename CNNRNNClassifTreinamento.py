###################
# Este componente executa o treino dos modelos de machine learning para recomendação por conteúdo
# com dados de entrada na forma de espectrogramas das músicas
# A entrada consiste nos espectrogramas de amostras de audio do Usuário A com a classe (0-não curte, 1-curte)
# Uso de validação cruzada (cross-validation) para cálculo das métricas.
# O modelo com melhor acurácia é salvo como modeloClassifEspectro.pickle
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
import statistics
from utils import calcula_cross_val_scores
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
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
from sklearn.metrics._classification import accuracy_score, classification_report
from tensorflow.python.keras.models import load_model
tf.get_logger().setLevel('ERROR') 

logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifTreinamentoCNNRNN')

#lendo datasets
npzTreino = np.load('./FeatureStore/AudioEspectrogramasTreinoBin.npz')
X_TreinVal= npzTreino['arr_0']
y_TreinVal = npzTreino['arr_1']

npzTeste = np.load('./FeatureStore/AudioEspectrogramasTesteBin.npz')
X_teste= npzTeste['arr_0']
y_teste = npzTeste['arr_1']

# dividindo treinamento em treino e validação
X_trein, X_val, y_trein, y_val = train_test_split(X_TreinVal, y_TreinVal, random_state=0, test_size=0.25)

#expande dimensões para uso por conv2d
X_trein = np.expand_dims(X_trein, axis = -1)
X_val = np.expand_dims(X_val, axis = -1)
X_teste = np.expand_dims(X_teste, axis = -1)
X_TreinVal = np.expand_dims(X_TreinVal, axis = -1)

def build_modelo():
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

# definindo o modelo, com os hiperparametros previamente escolhidos
model = build_modelo()
checkpoint = ModelCheckpoint('./FeatureStore/modeloClassifCNNRNN', monitor='val_accuracy', verbose=1,
                                save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=20)
reducelr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_delta=0.01,verbose=1)
callbacks_list = [reducelr, early_stop, checkpoint]

history = model.fit(X_trein, y_trein, batch_size=20, epochs=100,validation_data=(X_val, y_val), 
                    verbose=1, callbacks=callbacks_list)

# Cálculo de acurácia:
#
# y_predicao = model.predict (X_teste)
# y_predicao = [round(y[0]) for y in y_predicao]
# acuracia = accuracy_score (y_teste, y_predicao)
# print("acuracia CNNeRNN {:.3f}".format(acuracia))
# report = classification_report (y_teste, y_predicao, target_names=["Nao Curte", "Curte"])
# print(report)

model = load_model ('./FeatureStore/modeloClassifCNNRNN')
y_predicao = model.predict (X_teste)
y_predicao = [round(y[0]) for y in y_predicao]
acuracia = accuracy_score (y_teste, y_predicao)
report = classification_report (y_teste, y_predicao, target_names=["Nao Curte", "Curte"])
print("acuracia CNNeRNN {:.3f}".format(acuracia))
print("Resultados CNNeRNN:")
print(report)

# Treino do classificador com todos os dados
# e salvando o modelo treinado
model.fit(X_TreinVal, y_TreinVal, batch_size=20, epochs=100,validation_data=(X_teste, y_teste), 
                verbose=1, callbacks=callbacks_list)

logging.info ("acuracia CNNeRNN {:.3f}".format(acuracia))
logging.info ("report CNNeRNN:\n {}".format(report))

logging.info('\n<< ClassifTreinamentoCNNRNN')
