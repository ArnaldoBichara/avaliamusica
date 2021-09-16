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

# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

#!!!!!!
# Particionando a base de dados, 30% para teste
X_trein, X_teste, labels_trein, labels_teste = train_test_split(X, y, random_state=0, test_size=0.30)

scoring = ['balanced_accuracy','precision_macro', 'recall_macro', 'roc_auc', 'neg_mean_squared_error']

def calcula_cross_val_scores(clf, X_trein, labels_trein, scoring):
    mul_scores = cross_validate(clf, X_trein, labels_trein, scoring=scoring, return_train_score=True)
    acuracias_treino = mul_scores['train_balanced_accuracy']
#    print ("train_balanced_accuracy:      {:.2e} (+/- {:.2e})".format(acuracias_treino.mean(), acuracias_treino.std()))
    acuracias_teste  = mul_scores['test_balanced_accuracy']
#    print ("test_balanced_accuracy:       {:.2e} (+/- {:.2e})".format(acuracias_teste.mean(), acuracias_teste.std()))
    # scores = mul_scores['test_precision_macro']
    # print ("test_precision_macro:         {:.2e} (+/- {:.2e})".format(scores.mean(), scores.std()))
    # scores = mul_scores['test_recall_macro']
    # print ("test_recall_macro:            {:.2e} (+/- {:.2e})".format(scores.mean(), scores.std()))
    # scores = mul_scores['test_roc_auc']
    # print ("test_roc_auc:                 {:.2e} (+/- {:.2e})".format(scores.mean(), scores.std()))
    # scores = mul_scores['train_neg_mean_squared_error']
    # print ("train_neg_mean_squared_error: {:.2e} (+/- {:.2e})".format(-scores.mean(), scores.std()))
    # scores = mul_scores['test_neg_mean_squared_error']
    # print ("test_neg_mean_squared_error : {:.2e} (+/- {:.2e})".format(-scores.mean(), scores.std()))

#%%
# Análise modelo RandomForest
# Shallow decision trees (e.g. few levels) generally do not overfit but have poor performance (high bias, low variance). 
# Whereas deep trees (e.g. many levels) generally do overfit and have good performance (low bias, high variance). 
# A desirable tree is one that is not so shallow that it has low skill and not so deep that it overfits the training dataset.
acuracias_treino, acuracias_teste = list(), list()
profundidades = [i for i in range(1,21)]
for profundidade in profundidades: 
#    print ("\nRandom Forest com max_depth: {:d}".format(profundidade))
    # n_jobs= -1 : usa todos os processadores da máquina
    clf = RandomForestClassifier(max_depth=profundidade)
    #scores = cross_val_score (clf, X_trein, labels_trein, cv=10)
    mul_scores = cross_validate(clf, X_trein, labels_trein, scoring=scoring, return_train_score=True)
    acuracias_treino.append( mul_scores['train_balanced_accuracy'].mean() )
    acuracias_teste.append( mul_scores['test_balanced_accuracy'].mean() )
    print('>%d, treino: %.3f, teste: %.3f' % (profundidade, mul_scores['train_balanced_accuracy'].mean(), mul_scores['test_balanced_accuracy'].mean()))

pyplot.plot(profundidades, acuracias_treino, '-o', label='Treino')
pyplot.plot(profundidades, acuracias_teste, '-o', label='Teste')
pyplot.legend()
pyplot.show()

# adaBoost RandomForest
# print ("\nAdaBoost Random Forest:")
# clf = AdaBoostClassifier()
# calcula_cross_val_scores(clf, X_trein, labels_trein, scoring=scoring)
#%%
#adaBoost random forest cross validation 
# print ("\nAdaBoost Random Forest: max_depth=2")
# rf = RandomForestClassifier(max_depth=2)
# clf = AdaBoostClassifier(base_estimator=rf)
# calcula_cross_val_scores(clf, X_trein, labels_trein, scoring=scoring)

#%%
#adaBoost random forest cross validation 
# print ("\nAdaBoost Random Forest: max_depth=2")
# rf = RandomForestClassifier(max_depth=2)
# clf = AdaBoostClassifier(base_estimator=rf, n_estimators=200)
# calcula_cross_val_scores(clf, X_trein, labels_trein, scoring=scoring)
#%%
#adaBoost svc cross validation 
# print ("\nAdaBoost SVC:")
# from sklearn.svm import SVC
# svc=SVC(probability=True, kernel='linear')
# clf = AdaBoostClassifier(base_estimator=svc)
# calcula_cross_val_scores(clf, X_trein, labels_trein, scoring=scoring)


#%%
#logging.info('\n<< Classif Analise Treinamento')
