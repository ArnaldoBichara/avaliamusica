###################
# Faz busca automática de melhores hiperparametros 
# do RandomForest
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection._search import RandomizedSearchCV

logging.basicConfig(filename='./Analises/EscolhadeHiperparametros.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

#lendo dataset
npzfile = np.load('./FeatureStore/AudioFeaturesUserATreino.npz')
X = npzfile['arr_0']
y = npzfile['arr_1']

random_grid = {'n_estimators': [200, 300, 400, 500],
                'max_depth': [8,10,12,14],
               'min_samples_split': [2,3,4],
               'max_leaf_nodes': range(90,114, 4),
               'min_samples_leaf': [1, 2],
#               'max_samples': range(300,400,5)
               }


# Análise modelo RandomForest
# Shallow decision trees (e.g. few levels) generally do not overfit but have poor performance (high bias, low variance). 
# Whereas deep trees (e.g. many levels) generally do overfit and have good performance (low bias, high variance). 
# A desirable tree is one that is not so shallow that it has low skill and not so deep that it overfits the training dataset.
#%%
# cross validation tipo stratifiedKFole, com 3 repetições e 10 splits
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
clf = RandomForestClassifier()
clf_random = RandomizedSearchCV (estimator = clf, param_distributions = random_grid, cv = cv, verbose=1, n_jobs=-1)
search = clf_random.fit (X,y)
print (search.best_params_)
#%%
#logging.info('\n<< Classif Analise Treinamento')
