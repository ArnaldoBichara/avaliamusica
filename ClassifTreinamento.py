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
import statistics
from utils import calcula_cross_val_scores

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

import tensorflow as tf
from statistics import mean
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.layers.normalization import BatchNormalization
tf.get_logger().setLevel('ERROR') 


logging.basicConfig(filename='./Analises/processamentoClassificacao.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifTreinamento')

#lendo datasets
npzTreino = np.load('./FeatureStore/AudioFeaturesUserATreino.npz')
X_trein= npzTreino['arr_0']
y_trein = npzTreino['arr_1']
npzTeste = np.load('./FeatureStore/AudioFeaturesUserATeste.npz')
X_teste= npzTeste['arr_0']
y_teste = npzTeste['arr_1']

X = np.append (X_trein, X_teste, axis=0)
y = np.append (y_trein, y_teste, axis=0 )

def create_model_MLP(optimizer='adam', init='uniform', 
    weight_constraint=0, dropout_rate=0.1, kr = l2(0.001)):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='relu', 
            kernel_initializer=init, 
            kernel_regularizer=kr))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(9,  activation='relu', kernel_initializer=init,
            kernel_regularizer=kr))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init,
            kernel_regularizer=kr))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#%% definindo o modelo, com os hiperparametros previamente escolhidos
rf = RandomForestClassifier(n_estimators=400,
                            n_jobs=-1,
                            max_depth=10,
                            min_samples_split=4,
                            min_samples_leaf=1,
                            max_leaf_nodes=94)
base_estimator = DecisionTreeClassifier(max_depth=1)
ab = AdaBoostClassifier (n_estimators=400,
                         learning_rate=0.06,
                         base_estimator=base_estimator)    
gb = GradientBoostingClassifier (n_estimators=400,
                                learning_rate=0.08,
                                max_depth=1)  
mlp = Pipeline([
                ('standardize', StandardScaler()), # passa média para 0 e desvio para 1
                ('mlp', KerasClassifier(build_fn=create_model_MLP, epochs=100, batch_size=32, verbose=0))
            ])                                                       

# Cálculo de acurácia:
#
acuracias=[]
for i in range (5):
    rf.fit (X_trein, y_trein)
    y_predicao = rf.predict (X_teste)
    acuracia = np.sum(y_predicao == y_teste)/len(y_teste)
    acuracias.append(acuracia)
acuraciarf = mean(acuracias)
print("acuracia RandomForest {:.3f}".format(acuraciarf))
logging.info ("acuracia RandomForest {:.3f}".format(acuraciarf))

acuracias=[]
for i in range (5):
    ab.fit (X_trein, y_trein)
    y_predicao = ab.predict (X_teste)
    acuracia = np.sum(y_predicao == y_teste)/len(y_teste)
    acuracias.append(acuracia)
acuraciaab = mean(acuracias)
print("acuracia AdaBoost {:.3f}".format(acuraciaab))
logging.info ("acuracia AdaBoost {:.3f}".format(acuraciaab))

acuracias=[]
for i in range (5):
    gb.fit (X_trein, y_trein)
    y_predicao = gb.predict (X_teste)
    acuracia = np.sum(y_predicao == y_teste)/len(y_teste)
    acuracias.append(acuracia)
acuraciagb = mean(acuracias)
print("acuracia GradientBoost {:.3f}".format(acuraciagb))
logging.info ("acuracia GradientBoost {:.3f}".format(acuraciagb))

acuracias=[]
for i in range (5):
    mlp.fit (X_trein, y_trein)
    y_predicao = mlp.predict (X_teste)
    # os valores de y_predicao são entre 0 e 1 (probabilidade). 
    # Então temos de arredondá-los
    y_predicao = [round(y[0]) for y in y_predicao]
    acuracia = np.sum(y_predicao == y_teste)/len(y_teste)
    acuracias.append(acuracia)
acuraciamlp = mean(acuracias)
print("acuracia Rede Neural MLP {:.3f}".format(acuraciamlp))
logging.info ("acuracia Rede Neural MLP {:.3f}".format(acuraciamlp))
#
# Treino do classificador com todos os dados
# e salvando o modelo treinado
if (acuraciamlp > acuraciagb):
    if (acuraciamlp > acuraciaab):
        if (acuraciamlp > acuraciarf):
            modeloEscolhido = mlp
        else:
            modeloEscolhido = rf
    else:
        if (acuraciaab > acuraciarf):
            modeloEscolhido = ab
        else:
            modeloEscolhido = rf
else:
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

# montando o modelo com todos os dados
modeloEscolhido.fit (X, y)
with open("./FeatureStore/modeloClassif.pickle", 'wb') as arq:
    pickle.dump (modeloEscolhido, arq)

logging.info('\n<< ClassifTreinamento')
