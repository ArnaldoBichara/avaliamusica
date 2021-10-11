###################
# Análise de classificação por conteúdo,
# usando Redes Neurais
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import logging
from utils import calcula_cross_val_scores

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

logging.basicConfig(filename='./Analises/processamClassifNeuralNetwork.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifNeuralNetwork analise ')
#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")
#
# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

numDimEntrada = len(X.columns)

# modelo baseline 
# - número de neurônios igual ao número de entradas
# - ativação ReLu
# - uma saída com ativação sigmoid
def create_baseline():
	# create model
    model = Sequential()
    numNeuronios = numDimEntrada
    model.add(Dense(numNeuronios, input_dim=numDimEntrada, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#%% Estimador básico
logging.info('\nEstimador básico ')
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=50, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
logging.info("Acurácia: {:.2f}% ({:.2f}%)".format(results.mean()*100, results.std()*100))

# Estimador com normalização dos mini-batchs
logging.info('\nEstimador com normalização ')
estimators =[]
estimators.append(('standardize', StandardScaler())) # passa média para 0 e desvio para 1
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=50, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, y, cv=kfold)
logging.info("Acurácia: {:.2f}% ({:.2f}%)".format(results.mean()*100, results.std()*100))

#%% experimento com 9 neuronios ao invés de 12
# Essa diminuição é interessante quando há redundância nas variáveis de entrada
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=numDimEntrada, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
logging.info('\nEstimador com normalização e 9 neurônios')
estimators =[]
estimators.append(('standardize', StandardScaler())) # passa média para 0 e desvio para 1
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=50, verbose=0)))
pipeline = Pipeline(estimators)
#calcula_cross_val_scores (pipeline, X, y)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, y, cv=kfold)
logging.info("Acurácia: {:.2f}% ({:.2f}%)".format(results.mean()*100, results.std()*100))
#%% experimento: aumentando uma camada escondida, de forma a permitir
# melhor adaptação a funções não lineares
# 12 entradas -> [12 -> 6] -> 1 saída
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=numDimEntrada, activation='relu'))
    model.add(Dense(6,  activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
logging.info('\nEstimador com normalização e duas camadas escondidas')
estimators =[]
estimators.append(('standardize', StandardScaler())) # passa média para 0 e desvio para 1
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=50, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, y, cv=kfold)
logging.info("Acurácia: {:.2f}% ({:.2f}%)".format(results.mean()*100, results.std()*100))

#%% experimento com 6 neuronios ao invés de 12
# Essa diminuição é interessante quando há redundância nas variáveis de entrada
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=numDimEntrada, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
logging.info('\nEstimador com normalização e 6 neurônios')
estimators =[]
estimators.append(('standardize', StandardScaler())) # passa média para 0 e desvio para 1
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=50, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, y, cv=kfold)
logging.info("Acurácia: {:.2f}% ({:.2f}%)".format(results.mean()*100, results.std()*100))
#%%
logging.info('\n<< ClassifNeuralNetwork analise ')
