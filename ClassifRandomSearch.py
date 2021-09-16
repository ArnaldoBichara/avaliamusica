###################
# Faz busca automática de melhores hiperparametros 
# usando RandomizedSearchCV
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
from matplotlib import pyplot 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection._search import RandomizedSearchCV

logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")
#
# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

random_grid = {'max_depth': range (8,14),
               'min_samples_split': [2,3,4,5,6,8],
               'max_leaf_nodes': range(90,120, 4),
               'min_samples_leaf': [1, 2, 4],
#               'max_samples': range(300,400,5)
               }


# Análise modelo RandomForest
# Shallow decision trees (e.g. few levels) generally do not overfit but have poor performance (high bias, low variance). 
# Whereas deep trees (e.g. many levels) generally do overfit and have good performance (low bias, high variance). 
# A desirable tree is one that is not so shallow that it has low skill and not so deep that it overfits the training dataset.
#%%
clf = RandomForestClassifier()
clf_random = RandomizedSearchCV (estimator = clf, param_distributions = random_grid, n_iter = 2000, cv = 5, verbose=2, n_jobs=-1)
search = clf_random.fit (X,y)
print (search.best_params_)
#%%
#logging.info('\n<< Classif Analise Treinamento')
