###################
# Análise de treinamento RandomForest, seus hiperparâmetros
# os resultados obtidos e o uso de Adaboost
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> Analise treinam RandomForest')

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")

# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

# Particionando a base de dados, 25% para teste
X_trein, X_teste, labels_trein, labels_teste = train_test_split(X, y, random_state=0, test_size=0.25)
#%%
# modelo RandomForest
print ("Random Forest:")
logging.info ("Random Forest:")
# n_jobs= -1 : usa todos os processadores da máquina
clf = RandomForestClassifier()
scores = cross_val_score (clf, X_trein, labels_trein, cv=10)
mul_scores = cross_validate(clf, X_trein, labels_trein)
print (np.mean(scores))
print (mul_scores)
# # executa treinamento
# clf.fit(X_trein, labels_trein)
# # faz predição dos dados de teste
# labels_predicao = clf.predict(X_teste)
# matriz_confusao = confusion_matrix(labels_teste, labels_predicao)
# report = classification_report(labels_teste, labels_predicao)

# print (report)
# print (matriz_confusao)
# logging.info ("report %s", report)
# logging.info ("matriz de confusão %s", matriz_confusao)
#%%

# adaBoost RandomForest
print ("AdaBoost Random Forest:")
logging.info ("AdaBoost Random Forest:")
clf = AdaBoostClassifier(n_estimators=1000)
# executa treinamento
clf.fit(X_trein, labels_trein)
# faz predição dos dados de teste
labels_predicao = clf.predict(X_teste)
matriz_confusao = confusion_matrix(labels_teste, labels_predicao)
report = classification_report(labels_teste, labels_predicao)

print (report)
print (matriz_confusao)
logging.info ("report %s", report)
logging.info ("matriz de confusão %s", matriz_confusao)

#%%
#adaBoost random forest cross validation 
clf = AdaBoostClassifier()
scores = cross_val_score (clf, X_trein, labels_trein, cv=10)
print (np.mean(scores))

#%%
#adaBoost random forest cross validation 
rf = RandomForestClassifier(max_depth=2)
clf = AdaBoostClassifier(base_estimator=rf)
scores = cross_val_score (clf, X_trein, labels_trein, cv=10)
print (np.mean(scores))

#%%
#adaBoost svc cross validation 
from sklearn.svm import SVC
svc=SVC(probability=True, kernel='linear')
clf = AdaBoostClassifier(base_estimator=svc)
scores = cross_val_score (clf, X_trein, labels_trein)
print (np.mean(scores))

#%%
logging.info('\n<< Analise treinam RandomForest')
