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

# Particionando a base de dados, 30% para teste
X_trein, X_teste, labels_trein, labels_teste = train_test_split(X, y, random_state=0, test_size=0.30)


#%% treinando o modelo
acuracias_treino, acuracias_teste = list(), list()
clf = RandomForestClassifier(max_depth=10,
                             min_samples_split=3,
                             min_samples_leaf=2,
                             max_leaf_nodes=110)
calcula_cross_val_scores (clf, X_trein, labels_trein, cv=10)

# salvando modelo em .pickle
with open("./FeatureStore/modeloRandomForest.pickle", 'wb') as arq:
    pickle.dump (clf, arq)
#%%
logging.info('\n<< ClassifRandomForestTreinamento')
