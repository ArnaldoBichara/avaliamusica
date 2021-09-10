###################
# Este componente executa a predição entre Curtir ou Não Curtir uma música
# a partir do modelo contentbased RandomForest
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import sys

# Argumento de entrada 
# id_musica a prever
#%%
if (len(sys.argv)<1):
  print ('argumento obrigatório: id_musica')
  quit()
id_musica = int(sys.argv[1])

logging.basicConfig(filename='./Analises/Predição.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifPrediçãoRandomForestPredicao')

#lendo dataset
modeloRF = pd.read_pickle ("./FeatureStore/modeloRandomForest.pickle")

# buscando audiofeatures da música[id_musica] no spotify
TODO

# faz predição dos dados de teste
label_predicao = modeloRF.predict(dados_da_musica)
print(label_predicao)
logging.info ("predição %s", label_predicao)

#%%
logging.info('\n<< ClassifPrediçãoRandomForestPredicao')
