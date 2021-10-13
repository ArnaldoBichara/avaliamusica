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
tf.get_logger().setLevel('INFO') 

logging.basicConfig(filename='./Analises/EscolhadeHiperparametros.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

logging.info(">> ClassifRedeNeuralMLPEscolhadeHiperparametros")
#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")
#
# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])
numDimEntrada = len(X.columns)

# hiperparâmetros em teste
grid = { 'mlp__optimizer': ['rmsprop', 'adam'],
                'mlp__init': ['glorot_uniform', 'normal', 'uniform'],
                'mlp__epochs': [100, 200],
                'mlp__batch_size': [20, 32, 64],
                'mlp__dropout_rate': [0.0, 0.1, 0.2, 0.5, 0.7],
                'mlp__weight_constraint': [0, 1, 2, 3]} 

def create_model(optimizer='rmsprop', init='glorot_uniform', weight_constraint=0, dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=numDimEntrada, activation='relu', kernel_initializer=init, kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(9,  activation='relu', kernel_initializer=init))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

clf = Pipeline([
                ('standardize', StandardScaler()), # passa média para 0 e desvio para 1
                ('mlp', KerasClassifier(build_fn=create_model, verbose=0))
            ])
            
# cross validation tipo stratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True)

clf_grid = GridSearchCV (estimator = clf, param_grid = grid
                           , cv = kfold, verbose=2, n_jobs=-1)
#%%
search = clf_grid.fit (X,y)
print (search.best_params_, "acuracia:", search.best_score_)
logging.info ("{} Acurácia: {}".format(search.best_params_, search.best_score_))

logging.info("<< ClassifRedeNeuralMLPEscolhadeHiperparametros")

# %%
