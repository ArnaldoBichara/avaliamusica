###################
# Análise de classificação por conteúdo,
# usando Adaboost sobre árvore de decisão
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
from matplotlib import pyplot 
from utils import calcula_cross_val_scores

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
#logging.info('\n>> Classif Adaboost DT Analise')

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")
#
# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

#%% teste básico
clf = AdaBoostClassifier()
calcula_cross_val_scores (clf, X, y)

#%%
logging.info('\n<< Classif Adaboost DT Analise')
