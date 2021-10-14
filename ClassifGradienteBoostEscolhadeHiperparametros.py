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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection._search import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(filename='./Analises/EscolhadeHiperparametros.log', 
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



# hiperparâmetros em teste
random_grid = {'n_estimators': [300, 400,500,600],
            'learning_rate': [0.01, 0.04, 0.06, 0.08, 0.09, 0.1],
            'max_depth': [1] }
            

#%%
# cross validation tipo stratifiedKFole, com 3 repetições e 10 splits
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

clf = GradientBoostingClassifier()
clf_random = RandomizedSearchCV (estimator = clf, param_distributions = random_grid, n_iter = 2000, cv = cv, verbose=2, n_jobs=-1, random_state=1)
search = clf_random.fit (X,y)
print (search.best_params_, "acuracia:", search.best_score_)
logging.info('\n<< Classif Analise Treinamento')

# %%
