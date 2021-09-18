###################
# Este componente executa o treino dos modelos de machine learning para recomendação por conteúdo
# usando modelos RandomForest (bagging) e AdaBoost, GradienteBoosting (boosting)
# Os hiperparâmetros devem ser definidos previamente, através de Classif*EscolhadeHiperparametros.py
# A entrada consiste nas amostras de audio features do Usuário A com a classe (0-não curte, 1-curte)
# Uso de validação cruzada (cross-validation) para cálculo das métricas.
# O modelo com melhor acurácia é salvo como modeloClassif.pickle.
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
from utils import calcula_cross_val_scores

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
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
gb = GradientBoostingClassifier (n_estimators=400,
                                learning_rate=0.08)                           

X_trein, X_teste, y_trein, y_teste = train_test_split(X, y, random_state=0, test_size=0.30)
#
# Cálculo de acurácia:
#
rf.fit (X_trein, y_trein)
y_predicao = rf.predict (X_teste)
acuraciarf = np.sum(y_predicao == y_teste)/len(y_teste)
print("acurácia RandomForest:", acuraciarf)
logging.info ("acuracia RandomForest %s", acuraciarf)

ab.fit (X_trein, y_trein)
y_predicao = ab.predict (X_teste)
acuraciaab = np.sum(y_predicao == y_teste)/len(y_teste)
print("acurácia AdaBoost:", acuraciaab)
logging.info ("acuracia AdaBoost %s", acuraciaab)

gb.fit (X_trein, y_trein)
y_predicao = gb.predict (X_teste)
acuraciagb = np.sum(y_predicao == y_teste)/len(y_teste)
print("acurácia GradientBoost:", acuraciagb)
logging.info ("acuracia GradientBoost %s", acuraciagb)
#
# Treino do classificador com todos os dados
# e salvando o modelo treinado
if (acuraciagb > acuraciaab):
    if (acuraciagb > acuraciarf):
        modeloEscolhido = gb
    else:
        modeloEscolhido = rf
else:
    if (acuraciaab > acuraciarf):
        modeloEscolhido = ab
    else:
        modeloEscolhido = rf

modeloEscolhido.fit (X, y)
with open("./FeatureStore/modeloClassif.pickle", 'wb') as arq:
    pickle.dump (rf, arq)

logging.info('\n<< ClassifTreinamento')
