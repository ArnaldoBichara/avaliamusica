###################
# Faz busca automática de melhores hiperparametros 
# da rede neural MLP
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
from matplotlib import pyplot 
from utils import calcula_cross_val_scores

from sklearn.model_selection._search import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizers import rmsprop
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
tf.get_logger().setLevel('ERROR') 

logging.basicConfig(filename='./Analises/EscolhadeHiperparametros.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

logging.info(">> ClassifRedeNeuralMLPEscolhadeHiperparametros")

#lendo dataset
npzfile = np.load('./FeatureStore/AudioFeaturesUserATreino.npz')
X = npzfile['arr_0']
y = npzfile['arr_1']

# hiperparâmetros em teste
'''grid = { 'mlp__optimizer': ['rmsprop', 'adam', 'SGD', 'Adagrad'],
                'mlp__init': ['uniform'],
                'mlp__epochs': [100],
                'mlp__batch_size': [20],
                'mlp__dropout_rate': [0.1],
                'mlp__weight_constraint': [1]} '''
""" grid = { 'mlp__optimizer': ['rmsprop', 'adam'],
                'mlp__init': ['normal', 'uniform'],
                'mlp__epochs': [100],
                'mlp__batch_size': [20, 32],
                'mlp__dropout_rate': [0.1, 0.2],
                'mlp__weight_constraint': [1, 2]}  """
grid = { 'mlp__optimizer': ['adam'],
                'mlp__init': ['uniform'],
                'mlp__epochs': [600],
                'mlp__batch_size': [20],
                'mlp__dropout_rate': [0.4, 0.5, 0.6, 0.7]}                 

def create_model(optimizer='rmsprop', init='glorot_uniform', 
        weight_constraint=0, dropout_rate=0.0, kr = None):
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=12, activation='relu', 
            kernel_initializer=init, 
            kernel_regularizer=kr))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64,  activation='relu', kernel_initializer=init,
            kernel_regularizer=kr))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32,  activation='relu', kernel_initializer=init,
             kernel_regularizer=kr))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init,
            kernel_regularizer=kr))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, 
            metrics=['accuracy'])
    return model

early_stop = EarlyStopping(monitor='accuracy', patience=12)
#checkpoint=ModelCheckpoint('./FeatureStore/melhorModeloMLP', monitor='val_accuracy', save_best_only=True, mode='max', period=1)
reducelr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=8, min_delta=0.01,verbose=0)
keras_fit_params= {'mlp__callbacks': [reducelr, early_stop]}
pipeline = Pipeline([
                ('standardize', StandardScaler()), # passa média para 0 e desvio para 1
                ('mlp', KerasClassifier(build_fn=create_model, verbose=0))
            ])
            
# cross validation tipo stratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True)

#clf_random = RandomizedSearchCV (estimator = clf, param_distributions = grid,
#             n_iter = 100, cv = kfold, verbose=2, n_jobs=-1, random_state=1)
clf = GridSearchCV (estimator=pipeline, param_grid = grid, cv = kfold, verbose=2, n_jobs=-1)
search = clf.fit (X, y, mlp__callbacks=keras_fit_params)
#%%
print (search.best_params_, "acuracia:", search.best_score_)
logging.info ("{} Acurácia: {}".format(search.best_params_, search.best_score_))

logging.info("<< ClassifRedeNeuralMLPEscolhadeHiperparametros")

# %%
