# A partir de MusUsers (previamente filtrado) de MusUserA
# vamos montar o domínio das Musicas Colab, ou seja
# para análise colaborativa

#%% Importando packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Analises/preprocessamentoColab.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> FiltraDomMusicasColab')

musUsers        =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
musUserA        =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
musUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  

domMusColab = pd.concat([musUsers['id_musica'], musUserA['id_musica'], musUserANaoCurte['id_musica']], ignore_index=True, verify_integrity=True)        

domMusColab.drop_duplicates(inplace=True)

#%% removendo linhas que tenham algo com NaN
domMusColab=domMusColab.dropna()

# removendo linhas onde id_musica não é uma string válida (por algum motivo isso está acontecendo)
domMusColab= domMusColab[domMusColab.apply(lambda x: isinstance(x,str))]

domMusColab.reset_index(drop=True)

# ordenando dominioMusicas para tornar acesso mais rápido
domMusColab.sort_values(kind='quicksort', inplace=True)


#%%
logging.info ("Domínio de músicas para Colab =%s", domMusColab.shape)
# salvando dataset
domMusColab.to_pickle ("./FeatureStore/DominioMusicasColab.pickle")

logging.info('<< FiltraDomMusicasColab')

#

# %%
