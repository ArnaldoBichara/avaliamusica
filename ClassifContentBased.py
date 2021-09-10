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

logging.basicConfig(filename='./Analises/treinoModeloPorConteudo.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ConteudoTreina')

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")

# função que executa o treinamento da classificação
# e retorna predição dos dados de teste
def ExecutaClassificação(dados_treino, labels_treino, dados_teste):
    # njobs= -1 : usa todos os processadores da máquina
    clf = RandomForestClassifier(njobs=-1)
    clf.fit(dados_treino, labels_treino)
    return clf.predict(dados_teste)

#%% dividindo dataset em folds, para uso pelo procedimento de cross-validation
num_folds = 5

X_folds = np.array_split ( UserAFeatureSamples.drop(columns=['classe']) , num_folds)
y_folds = np.array_split ( np.array(UserAFeatureSamples['classe'])      , num_folds)

#%% treinando o modelo
acuracias=[]
for i in range(num_folds):  
    dados_trein   = np.concatenate (X_folds[:i] + X_folds[i+1:])
    labels_trein = np.concatenate (y_folds[:i] + y_folds[i+1:])
    dados_teste   = X_folds[i]
    labels_teste = y_folds[i]
    labels_predicao = ExecutaClassificação (dados_trein, labels_trein, dados_teste)
    acuracia = np.sum(labels_predicao == labels_teste)/len(labels_teste)
    acuracias.append(acuracia)


#%%
logging.info('\n<< ConteudoTreina')
