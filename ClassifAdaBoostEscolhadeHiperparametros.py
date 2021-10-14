###################
# Faz busca automática de melhores hiperparametros 
# do Adaboost
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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection._search import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

dt1 = DecisionTreeClassifier(max_depth=1)
dt2 = DecisionTreeClassifier(max_depth=2)
svc = SVC( probability= True, kernel='linear')

# hiperparâmetros em teste
random_grid = {'n_estimators': [200, 300, 400],
            'learning_rate': [0.01, 0.03, 0.05, 0.06, 0.1],
            'base_estimator': [dt1, dt2] }
random_grid2 = {'n_estimators': [50],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                'base_estimator': [svc] }            

#%%
# cross validation tipo stratifiedKFole, com 3 repetições e 10 splits
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

# clf = AdaBoostClassifier()
# clf_random = RandomizedSearchCV (estimator = clf, param_distributions = random_grid2, cv = cv, verbose=2, n_jobs=-1)
# search = clf_random.fit (X,y)
# print (search.best_params_, "acuracia:", search.best_score_)

clf = AdaBoostClassifier()
clf_random = RandomizedSearchCV (estimator = clf, param_distributions = random_grid, cv = cv, verbose=1, n_jobs=-1, random_state=1)
search = clf_random.fit (X,y)
print (search.best_params_, "acuracia:", search.best_score_)
#%% Para uma condição n_estimator e learning_rate dada, vamos verifica como se comporta o max_depth das árvores
# scores_treino = list()
# scores_teste = list()
# for i in range (1,7):
#     base = DecisionTreeClassifier(max_depth=i)
#     clf = AdaBoostClassifier (base_estimator=base, n_estimators=500, learning_rate=0.06)
#     acuracia_treino, acuracia_teste = calcula_cross_val_scores(clf, X, y)
#     scores_treino.append (acuracia_treino)
#     scores_teste.append(acuracia_teste)
# #
# pyplot.plot(range(1,7), scores_treino, '-o', label='Treino')
# pyplot.plot(range(1,7), scores_teste, '-o', label='Teste')
# pyplot.legend()
# pyplot.show()

logging.info('\n<< Classif Analise Treinamento')

# %%
