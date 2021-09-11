###################
# Este componente executa o treino dos modelos de machine learning para recomendação por conteúdo
# A entrada é as amostras de audio features do Usuário A com a classe (0-não curte, 1-curte)
# Uso de validação cruzada (cross-validation)
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifRandomForestTreinamento')

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")

#%% dividindo dataset em folds, para uso pelo procedimento de cross-validation
num_folds = 10

X_folds = np.array_split ( UserAFeatureSamples.drop(columns=['classe']) , num_folds)
y_folds = np.array_split ( np.array(UserAFeatureSamples['classe'])      , num_folds)

#%% treinando o modelo
acuracias=[]
# n_jobs= -1 : usa todos os processadores da máquina
modeloRF = RandomForestClassifier(n_jobs=-1)
for i in range(num_folds):  
    dados_trein   = np.concatenate (X_folds[:i] + X_folds[i+1:])
    labels_trein = np.concatenate (y_folds[:i] + y_folds[i+1:])
    dados_teste   = X_folds[i]
    labels_teste = y_folds[i]
    # executa treinamento
    modeloRF.fit(dados_trein, labels_trein)
    # faz predição dos dados de teste
    labels_predicao = modeloRF.predict(dados_teste)
    acuracia = np.sum(labels_predicao == labels_teste)/len(labels_teste)
    acuracias.append(acuracia)
print(acuracias)
logging.info ("acuracias %s", acuracias)

# salvando modelo em .pickle
with open("./FeatureStore/modeloRandomForest.pickle", 'wb') as arq:
    pickle.dump (modeloRF, arq)
#%%
logging.info('\n<< ClassifRandomForestTreinamento')
