###################
# Análise de treinamento da classificacao por conteúdo, seus hiperparâmetros
# os resultados obtidos e o uso de Adaboost
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
from matplotlib import pyplot 
from utils import calcula_cross_val_scores

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
#logging.info('\n>> Classif Analise Treinamento')

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")
#
# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

#!!!!!!
# Particionando a base de dados, 30% para teste
X_trein, X_teste, labels_trein, labels_teste = train_test_split(X, y, random_state=0, test_size=0.30)

scoring = ['balanced_accuracy','precision_macro', 'recall_macro', 'roc_auc', 'neg_mean_squared_error']


#%% Calculando o RandoForest com os parametros definidos pelo RandomSearch
acuracias_treino, acuracias_teste = list(), list()
clf = RandomForestClassifier(max_depth=10,
                             min_samples_split=3,
                             min_samples_leaf=2,
                             max_leaf_nodes=110)
acuracia_treino, acuracia_teste = calcula_cross_val_scores (clf, X_trein, labels_trein, cv=10)

#%%

# Análise modelo RandomForest
# Shallow decision trees (e.g. few levels) generally do not overfit but have poor performance (high bias, low variance). 
# Whereas deep trees (e.g. many levels) generally do overfit and have good performance (low bias, high variance). 
# A desirable tree is one that is not so shallow that it has low skill and not so deep that it overfits the training dataset.
#%%
acuracias_treino, acuracias_teste = list(), list()
params = [i for i in range(1,21)]
for profundidade in params: 
#    print ("\nRandom Forest com max_depth: {:d}".format(profundidade))
    clf = RandomForestClassifier(max_depth=profundidade, min_samples_split=10)
    acuracia_treino, acuracia_teste = calcula_cross_val_scores (clf, X_trein, labels_trein, cv=5)
    acuracias_treino.append( acuracia_treino )
    acuracias_teste.append( acuracia_teste )
    print('>%d, treino: %.3f, teste: %.3f' % (profundidade, acuracia_treino, acuracia_teste))

pyplot.plot(params, acuracias_treino, '-o', label='Treino')
pyplot.plot(params, acuracias_teste, '-o', label='Teste')
pyplot.legend()
pyplot.show()

# com max-depth = 8 vamos alterar min_sample_split
#%%
acuracias_treino, acuracias_teste = list(), list()
params = [i for i in range(2,20)]
for param in params: 
#    print ("\nRandom Forest com max_depth: {:d}".format(profundidade))
    clf = RandomForestClassifier(max_depth=9, 
                                 min_samples_split=param)
    acuracia_treino, acuracia_teste = calcula_cross_val_scores (clf, X_trein, labels_trein, cv=5)
    acuracias_treino.append( acuracia_treino )
    acuracias_teste.append( acuracia_teste )
    print('>%d, treino: %.3f, teste: %.3f' % (param, acuracia_treino, acuracia_teste))

pyplot.plot(params, acuracias_treino, '-o', label='Treino')
pyplot.plot(params, acuracias_teste, '-o', label='Teste')
pyplot.legend()
pyplot.show()

#%%
acuracias_treino, acuracias_teste = list(), list()
params = [i for i in range(10,200,2)]
for param in params: 
#    print ("\nRandom Forest com max_depth: {:d}".format(profundidade))
    clf = RandomForestClassifier(max_depth=9, 
                                 max_leaf_nodes=param)
    acuracia_treino, acuracia_teste = calcula_cross_val_scores (clf, X_trein, labels_trein, cv=5)
    acuracias_treino.append( acuracia_treino )
    acuracias_teste.append( acuracia_teste )
    print('>%d, treino: %.3f, teste: %.3f' % (param, acuracia_treino, acuracia_teste))

pyplot.plot(params, acuracias_treino, '-o', label='Treino')
pyplot.plot(params, acuracias_teste, '-o', label='Teste')
pyplot.legend()
pyplot.show()

#%%
acuracias_treino, acuracias_teste = list(), list()
params = [i for i in range(1,100,5)]
for param in params: 
#    print ("\nRandom Forest com max_depth: {:d}".format(profundidade))
    clf = RandomForestClassifier(max_depth=9, 
                                 min_samples_leaf=param)
    acuracia_treino, acuracia_teste = calcula_cross_val_scores (clf, X_trein, labels_trein, cv=5)
    acuracias_treino.append( acuracia_treino )
    acuracias_teste.append( acuracia_teste )
    print('>%d, treino: %.3f, teste: %.3f' % (param, acuracia_treino, acuracia_teste))

pyplot.plot(params, acuracias_treino, '-o', label='Treino')
pyplot.plot(params, acuracias_teste, '-o', label='Teste')
pyplot.legend()
pyplot.show()

#%%
acuracias_treino, acuracias_teste = list(), list()
params = [i for i in range(1,100,5)]
for param in params: 
#    print ("\nRandom Forest com max_depth: {:d}".format(profundidade))
    clf = RandomForestClassifier(max_depth=9, 
                                 min_samples_leaf=param)
    acuracia_treino, acuracia_teste = calcula_cross_val_scores (clf, X_trein, labels_trein, cv=5)
    acuracias_treino.append( acuracia_treino )
    acuracias_teste.append( acuracia_teste )
    print('>%d, treino: %.3f, teste: %.3f' % (param, acuracia_treino, acuracia_teste))

pyplot.plot(params, acuracias_treino, '-o', label='Treino')
pyplot.plot(params, acuracias_teste, '-o', label='Teste')
pyplot.legend()
pyplot.show()

#%%
#logging.info('\n<< Classif Analise Treinamento')
