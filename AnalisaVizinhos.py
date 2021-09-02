# vamos analisar como é essa base de user
# pergunta: qual a distribuição de músicas por user?

#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Analises/preprocessamentoColab.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> Analisa Vizinhos')

MusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
MusUserA =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
MusUserAbarra =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  

def AchaMusIguais (userid, MusUserA):
    MusViz = MusUsers[MusUsers['userid']==userid]['interpretacao'].array
    UserA = MusUserA['interpretacao'].array
    return MusViz[np.where(np.in1d(MusViz, UserA, assume_unique=True))]

#%% Vizinho mais próximo de A: 4112a2d73341b083607be0dbb918a9a0
mus = AchaMusIguais ('4112a2d73341b083607be0dbb918a9a0', MusUserA)
mus
# bateu!! São 27 músicas, em geral da Enya

#%% 2o Vizinho mais próximo de A: bdaa31836fbca8c96f923d238d9f7635
mus = AchaMusIguais ('bdaa31836fbca8c96f923d238d9f7635', MusUserA)
mus
# bateu.São 18 músicas, em geral da Enya
#%% 3o Vizinho: ec4ee589606b6698f0703f5c28c015ad
mus = AchaMusIguais ('ec4ee589606b6698f0703f5c28c015ad', MusUserA)
mus
# ué, tem 20 músicas iguais e está abaixo do 2o. 
# se um user tem mais músicas, a distância aumenta
#%%
#%%
logging.info('<< Analisa Vizinhos')

# %%
