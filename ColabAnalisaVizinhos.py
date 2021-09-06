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

VizinhosUserA      = VizinhosUserA[:15]
VizinhosUserAbarra = VizinhosUserAbarra[:15]

# Lendo matrizes esparsas
dfEsparsaUsers      =  pd.read_pickle ("./FeatureStore/ColabMusUsersEsparsa.pickle")  
dfEsparsaUserA      = pd.read_pickle("./FeatureStore/ColabMusUserAEsparsa.pickle")
dfEsparsaUserAbarra = pd.read_pickle("./FeatureStore/ColabMusUserAbarraEsparsa.pickle")

#%%
userid = VizinhosUserA.iloc[0]['userid']
dfEsparsaVizinho = dfEsparsaUsers[dfEsparsaUsers['user']==VizinhosUserA.iloc[0]['userid']]
#%%
res = confusion_matrix (dfEsparsaUserA, VizinhosUserA[0])
 
#%%
logging.info('<< Analisa Vizinhos')

# %%
