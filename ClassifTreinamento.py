###################
# Este componente executa o treino dos modelos de machine learning para recomendação por conteúdo
# usando modelos RandomForest (bagging) e AdaBoost (boosting)
# Os hiperparâmetros foram definidos previamente, em outro módulo
# A entrada é as amostras de audio features do Usuário A com a classe (0-não curte, 1-curte)
# Uso de validação cruzada (cross-validation)
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
from utils import calcula_cross_val_scores

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split


logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifTreinamento')

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")

# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

#%% definindo o modelo, com os hiperparametros previamente escolhidos
rf = RandomForestClassifier(n_jobs=-1,
                            max_depth=10,
                            min_samples_split=3,
                            min_samples_leaf=2,
                            max_leaf_nodes=110)
ab = AdaBoostClassifier (n_estimators=300,
                         learning_rate=0.1)                            

X_trein, X_teste, y_trein, y_teste = train_test_split(X, y, random_state=0, test_size=0.30)
#
# Cálculo de acurácia:
#
rf.fit (X_trein, y_trein)
y_predicao = rf.predict (X_teste)
acuracia = np.sum(y_predicao == y_teste)/len(y_teste)
print("acurácia RandomForest:", acuracia)
logging.info ("acuracia RandomForest %s", acuracia)

ab.fit (X_trein, y_trein)
y_predicao = ab.predict (X_teste)
acuracia = np.sum(y_predicao == y_teste)/len(y_teste)
print("acurácia AdaBoost:", acuracia)
logging.info ("acuracia AdaBoost %s", acuracia)
#
# Treino do classificador com todos os dados
# e salvando o modelo treinado

rf.fit (X, y)
with open("./FeatureStore/modeloRandomForest.pickle", 'wb') as arq:
    pickle.dump (rf, arq)

ab.fit (X, y)
with open("./FeatureStore/modeloAdaBoost.pickle", 'wb') as arq:
    pickle.dump (ab, arq)
#
logging.info('\n<< ClassifTreinamento')
