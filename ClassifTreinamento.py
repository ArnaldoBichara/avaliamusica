###################
# Este componente executa o treino dos modelos de machine learning para recomendação por conteúdo
# usando modelos RandomForest (bagging) e AdaBoost, GradienteBoosting (boosting)
# Os hiperparâmetros devem ser definidos previamente, através de Classif*EscolhadeHiperparametros.py
# A entrada consiste nas amostras de audio features do Usuário A com a classe (0-não curte, 1-curte)
# Uso de validação cruzada (cross-validation) para cálculo das métricas.
# O modelo com melhor acurácia é salvo como modeloClassif.pickle
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
from sklearn.metrics import balanced_accuracy_score

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
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.models import load_model
from sklearn.svm import SVC
import joblib
import os
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
    weight_constraint=0, dropout_rate=0.5, kr = None):
    # create model
    model = Sequential()
    # model.add(Dense(256, input_dim=12, activation='relu', 
    #         kernel_initializer=init, 
    #         kernel_regularizer=kr))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropout_rate))
    model.add(Dense(32,  activation='relu', kernel_initializer=init,
            kernel_regularizer=kr))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(12,  activation='relu', kernel_initializer=init,
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
                            max_depth=12,
                            min_samples_split=3,
                            min_samples_leaf=1,
                            max_leaf_nodes=98,
                            max_samples=None)
base_estimator = DecisionTreeClassifier(max_depth=1)
base_estimator2 = SVC( probability= True, kernel='linear')
ab = AdaBoostClassifier (n_estimators=400,
                         learning_rate=0.08,
                         base_estimator=base_estimator)    
ab2 = AdaBoostClassifier (n_estimators=400,
                         learning_rate=0.08,
                         base_estimator=base_estimator2)                                 
gb = GradientBoostingClassifier (n_estimators=400,
                                learning_rate=0.04,
                                max_depth=2)  

early_stop = EarlyStopping(monitor='val_accuracy', patience=50)
checkpoint=ModelCheckpoint('./FeatureStore/modeloClassif.h5', monitor='val_accuracy', save_best_only=True, mode='max')
reducelr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20, min_delta=0.01,verbose=0)
mlp_keras_fit_params= {'mlp__callbacks': [early_stop, reducelr, checkpoint]}
pipeline = Pipeline([
                ('standardize', StandardScaler()), # passa média para 0 e desvio para 1
                ('mlp', KerasClassifier(build_fn=create_model_MLP, epochs=600, batch_size=32, verbose=0))
            ])                                                       

# Cálculo de acurácia:
#
acuracias=[]
for i in range (5):
    rf.fit (X_trein, y_trein)
    y_predicao = rf.predict (X_teste)
    acuracia = balanced_accuracy_score (y_teste, y_predicao)
    acuracias.append(acuracia)
acuraciarf = mean(acuracias)
print("acuracia RandomForest {:.3f}".format(acuraciarf))
logging.info ("acuracia RandomForest {:.3f}".format(acuraciarf))

acuracias=[]
for i in range (5):
    ab.fit (X_trein, y_trein)
    y_predicao = ab.predict (X_teste)
    acuracia = balanced_accuracy_score (y_teste, y_predicao)
    acuracias.append(acuracia)
acuraciaab = mean(acuracias)
print("acuracia AdaBoost {:.3f}".format(acuraciaab))
logging.info ("acuracia AdaBoost {:.3f}".format(acuraciaab))

acuracias=[]
for i in range (1):
    ab2.fit (X_trein, y_trein)
    y_predicao = ab2.predict (X_teste)
    acuracia = balanced_accuracy_score (y_teste, y_predicao)
    acuracias.append(acuracia)
acuraciaab = mean(acuracias)
print("acuracia AdaBoost com SVC {:.3f}".format(acuraciaab))
logging.info ("acuracia AdaBoost SVC {:.3f}".format(acuraciaab))


acuracias=[]
for i in range (5):
    gb.fit (X_trein, y_trein)
    y_predicao = gb.predict (X_teste)
    acuracia = balanced_accuracy_score (y_teste, y_predicao)
    acuracias.append(acuracia)
acuraciagb = mean(acuracias)
print("acuracia GradientBoost {:.3f}".format(acuraciagb))
logging.info ("acuracia GradientBoost {:.3f}".format(acuraciagb))

X_trein, X_valid, y_trein, y_valid = train_test_split(X_trein, y_trein, random_state=0, test_size=0.25)

history = pipeline.fit (X_trein, y_trein, mlp__callbacks=mlp_keras_fit_params, mlp__validation_data=(X_valid, y_valid))

#y_predicao = pipeline.predict (X_teste)
## os valores de y_predicao são entre 0 e 1 (probabilidade), então temos de arredondá-los
#y_predicao = [round(y[0]) for y in y_predicao]
#acuraciamlp = balanced_accuracy_score (y_teste, y_predicao)
#print("acuracia Rede Neural imediata MLP {:.3f}".format(acuraciamlp))
#logging.info ("acuracia Rede imediata Neural MLP {:.3f}".format(acuraciamlp))

# carregando o melhor modelo mlp
pipeline.named_steps['mlp'].model = load_model('./FeatureStore/modeloClassif.h5')
ybest_predicao = pipeline.predict (X_teste)
# os valores de y_predicao são entre 0 e 1 (probabilidade), então temos de arredondá-los
ybest_predicao = [round(y[0]) for y in ybest_predicao]
acuraciamlp = balanced_accuracy_score (y_teste, ybest_predicao)
print("acuracia Rede Neural MLP {:.3f}".format(acuraciamlp))
logging.info ("acuracia Rede Neural MLP {:.3f}".format(acuraciamlp))

#
# Treino do classificador com todos os dados
# e salvando o modelo treinado
if ( (acuraciamlp > acuraciagb) and
     (acuraciamlp > acuraciaab) and
     (acuraciamlp > acuraciarf) ):
        # nesse caso temos de salvar o modelo mlp
        pipeline.fit (X, y)
        pipeline.named_steps['mlp'].model.save('./FeatureStore/modeloClassif.h5')
        pipeline.named_steps['mlp'].model = None
        joblib.dump(pipeline, './FeatureStore/modeloClassif.pickle')
else:
    # Limpando antigo modeloMPL, se existir
    if os.path.exists('./FeatureStore/modeloClassif.h5'):
        os.remove('./FeatureStore/modeloClassif.h5')
    if (acuraciaab > acuraciarf):
        if (acuraciaab > acuraciagb):
            modeloEscolhido = ab
        else:
            if (acuraciagb > acuraciarf):
                modeloEscolhido = gb
            else:
                modeloEscolhido = rf
    else:
        modeloEscolhido = rf            
    modeloEscolhido.fit (X, y)
    with open("./FeatureStore/modeloClassif.pickle", 'wb') as arq:
        pickle.dump (modeloEscolhido, arq)

logging.info('\n<< ClassifTreinamento')
