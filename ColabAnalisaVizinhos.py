# Vamos analisar a base de vizinhos, a matriz de confusão, se o resultado foi bom.

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
logging.info('>> Analisa Vizinhos')

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



#%%
logging.info('<< Analisa Vizinhos')

# %%
