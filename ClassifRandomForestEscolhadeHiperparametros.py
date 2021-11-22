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
from sklearn import metrics 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
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

grid =         {'n_estimators': [200, 300, 400, 500],
                'max_depth': [8,10,12,14],
               'min_samples_split': [2,3,4],
               'max_leaf_nodes': range(90,102, 4),
               'min_samples_leaf': [1, 2],
#               'max_samples': range(300,400,5)
               }


# Análise modelo RandomForest
# Shallow decision trees (e.g. few levels) generally do not overfit but have poor performance (high bias, low variance). 
# Whereas deep trees (e.g. many levels) generally do overfit and have good performance (low bias, high variance). 
# A desirable tree is one that is not so shallow that it has low skill and not so deep that it overfits the training dataset.
#%%
# cross validation tipo stratifiedKFold
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
cv = StratifiedKFold(n_splits=10)
clf = RandomForestClassifier()
clf_random = GridSearchCV ( estimator = clf, 
                            param_grid = grid,
                            cv = cv,
                            verbose=2,
                            n_jobs=-1,
                            scoring='balanced_accuracy')
search = clf_random.fit (X,y)
print (search.best_params_)
#%%
#logging.info('\n<< Classif Analise Treinamento')
