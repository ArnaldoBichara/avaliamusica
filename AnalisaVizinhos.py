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

domMus =  pd.read_pickle ("./FeatureStore/DominioMusicasColab.pickle")  


def AchaMusIguais (userid, MusUserA):
    MusViz = MusUsers[MusUsers['userid']==userid]['interpretacao'].array
    UserA = MusUserA['interpretacao'].array
    return MusViz[np.where(np.in1d(MusViz, UserA, assume_unique=True))]

#%% Vizinho mais próximo de A: 4112a2d73341b083607be0dbb918a9a0
mus = AchaMusIguais ('4112a2d73341b083607be0dbb918a9a0', MusUserA)
mus
# bateu!! São 27 músicas, em geral da Enya

#%% 2o Vizinho mais próximo de A: ec4ee589606b6698f0703f5c28c015ad
mus = AchaMusIguais ('ec4ee589606b6698f0703f5c28c015ad', MusUserA)
mus
# bateu.São 20 músicas, em geral da Enya
#%% 3o Vizinho: bdaa31836fbca8c96f923d238d9f7635
mus = AchaMusIguais ('bdaa31836fbca8c96f923d238d9f7635', MusUserA)
mus
# 20 musicas iguais. Na matriz de confusao indica apenas 18
#%%
print (MusUsers[MusUsers['interpretacao'].str.contains("Milton Nascimento", na= False, case=False)]['id_musica','interpretacao'].to_string(index=False))

#%%
print (MusUsers[MusUsers['interpretacao'].str.contains("MILTON NASCIMENTO:>TUDO O QUE VOCÊ PODIA SER", na= False, case=False)][['userid','id_musica']].to_string(index=False))
#%%
print (MusUserA[MusUserA['interpretacao'].str.contains("MILTON NASCIMENTO:>TU", na= False, case=False)][['id_musica','interpretacao']].to_string(index=False))
#%%
print (MusUsers[MusUsers['interpretacao'].str.contains("MILTON NASCIMENTO", na= False, case=False)][['userid','id_musica','interpretacao']].sort_values(by='userid').to_string(index=False))

#%% usuário que gosta ao menos de uma música do Milto, mas não está nos meus vizinhos: e04bbd1370cf4fff619aa8e740828f4d
mus = AchaMusIguais ('e04cafcfa5dbb166640ea4a1ade73d75', MusUserA)
mus
# 
#%%
print (domMus[domMus['interpretacao'].str.contains("MILTON NAS", na=False, case=False)][['interpretacao']].to_string(index=False))

#%%
logging.info('<< Analisa Vizinhos')

# %%
