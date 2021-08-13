##########################
# Inclui UserA e UserAbarrado (o 'oposto' do gosto do UserA)
# no dataset de Users e músicas que curte
##########################
# #%%
import pandas as pd
import numpy as np
import pickle
import logging
from time import gmtime, strftime

logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('IncluUserA_em_Users >>')

# 
dfUserA_MusCurte    =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
dfUserA_MusNaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  
df600kMusCurtem    =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  

# removendo coluna que não é usada em users
dfUserA_MusCurte = dfUserA_MusCurte.drop(columns=['id_musica'])
dfUserA_MusNaoCurte = dfUserA_MusNaoCurte.drop(columns='id_musica')
#%%
# %% Incluindo user A e user Abarrado no dataset User_MusCurtem
df600kMusCurtem = pd.concat([df600kMusCurtem, dfUserA_MusCurte, dfUserA_MusNaoCurte], ignore_index=True, verify_integrity=True)        

# caso rode esse componente duas vezes seguidas
df600kMusCurtem.drop_duplicates(inplace=True)

# %%
df600kMusCurtem.to_pickle('./FeatureStore/MusUsers.pickle')

logging.info ('IncluUserA_em_Users shape = %s', df600kMusCurtem.shape)


logging.info('IncluUserA_em_Users <<')
