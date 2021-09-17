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
from utils import calcula_cross_val_scores

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifRandomForestTreinamento')

#lendo dataset
UserAFeatureSamples = pd.read_pickle ("./FeatureStore/UserAFeatureSamples.pickle")

# X - features
# y - classe
X = UserAFeatureSamples.drop(columns=['classe'])
y = np.array(UserAFeatureSamples['classe'])

#%% definindo o modelo, com os hiperparametros previamente escolhidos
clf = RandomForestClassifier(n_jobs=-1,
                             max_depth=10,
                             min_samples_split=3,
                             min_samples_leaf=2,
                             max_leaf_nodes=110)
#
# Cálculo de acurácia:
#
X_trein, X_teste, y_trein, y_teste = train_test_split(X, y, random_state=0, test_size=0.30)
clf.fit (X_trein, y_trein)
y_predicao = clf.predict (X_teste)
acuracia = np.sum(y_predicao == y_teste)/len(y_teste)
print(acuracia)
logging.info ("acuracia %s", acuracia)
#
# Treino do classificador com todos os dados
# e salvando o modelo treinado

clf.fit (X, y)
with open("./FeatureStore/modeloRandomForest.pickle", 'wb') as arq:
    pickle.dump (clf, arq)
#
logging.info('\n<< ClassifRandomForestTreinamento')
