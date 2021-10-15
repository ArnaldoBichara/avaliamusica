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

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

logging.basicConfig(filename='./Analises/processamClassifNeuralNetwork.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifNeuralNetworkCNN analise ')
#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")
#
# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

numDimEntrada = len(X.columns)

#%%
logging.info('\n<< ClassifNeuralNetworkCNN analise ')
