# A partir da totalidade de vizinhos por faixa,
# encontramos os N melhores vizinhos do UserA (e do UserAbarra)
# e descobrimos as músicas candidatas 
#   (o conjunto das músicas desses usuários que ainda não fazem parte do UserA)
# salvamos as músicas candidatas em .pickle

#%% Importando packages
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import confusion_matrix

#
logging.basicConfig(filename='./Analises/preprocessamentoColab.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ColabMontaMusCandidatas')

VizinhosUserA       =  pd.read_pickle ("./FeatureStore/ColabVizinhosUserA.pickle")  
VizinhosUserAbarra  =  pd.read_pickle ("./FeatureStore/ColabVizinhosUserAbarra.pickle")  

# ordenando vizinhos por distancia e pegando apenas os 15 primeiros
VizinhosUserA.sort_values(by=['distancia'], inplace=True)
VizinhosUserAbarra.sort_values(by=['distancia'], inplace=True)

# Pegando apenas primeiros 15 users#%%
VizinhosUserA.reset_index(inplace=True, drop=True)
VizinhosUserAbarra.reset_index(inplace=True, drop=True)

NVizinhos = 15
VizinhosUserA      = VizinhosUserA[:NVizinhos]
logging.info ('\n%s Melhores vizinhos de UserA:', NVizinhos)
for i in range(NVizinhos):
    logging.info ("%s", VizinhosUserA.iloc[i]['userid'])

VizinhosUserAbarra = VizinhosUserAbarra[:NVizinhos]
logging.info ('\n%s Melhores vizinhos de UserAbarra:', NVizinhos)
for i in range(NVizinhos):
    logging.info ("%s", VizinhosUserAbarra.iloc[i]['userid'])

# Para cada user vizinho, 
#    encontra as músicas do User
#    'appenda' lista de músicas candidatas
#    'appenda' lista de músicas comuns com user A
# remove duplicates
# salva .pickle

listaMusCandidatas

#%%
logging.info('\n<< ColabMontaMusCandidatas')

# %%
