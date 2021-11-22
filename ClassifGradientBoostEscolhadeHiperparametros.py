###################
# Faz busca automática de melhores hiperparametros 
# do Gradientboost
# o GradientBoost parece ser mais efetivo que
# o AdaBoost quando temos outliers
# usando RandomizedSearchCV
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
from matplotlib import pyplot 
from utils import calcula_cross_val_scores

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection._search import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(filename='./Analises/EscolhadeHiperparametros.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

#lendo dataset
npzfile = np.load('./FeatureStore/AudioFeaturesUserATreino.npz')
X = npzfile['arr_0']
y = npzfile['arr_1']

# hiperparâmetros em teste
grid = {'n_estimators': [400],
            'learning_rate': [0.01, 0.04, 0.06, 0.08, 0.09, 0.1],
            'max_depth': [1, 2, 3, 4, None] }
            

#%%
# cross validation tipo stratifiedKFole
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)

clf = GradientBoostingClassifier()
clf_random = GridSearchCV ( estimator = clf,
                            param_grid=grid,
                            cv = cv,
                            verbose=2,
                            n_jobs=-1,
                            scoring='balanced_accuracy')
search = clf_random.fit (X,y)
print (search.best_params_, "acuracia:", search.best_score_)
logging.info('\n<< Classif Analise Treinamento')

# %%
