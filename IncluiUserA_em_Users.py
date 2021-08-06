##########################
# Inclui UserA e UserAbarrado (o 'oposto' do gosto do UserA)
# no dataset de Users e músicas que curte
##########################
# #%%
import pandas as pd
import numpy as np
import pickle
# 
dfUserA_MusCurte    =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
dfUserA_MusNaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  
df600kMusCurtem    =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  

# removendo coluna que não é usada em users
dfUserA_MusCurte = dfUserA_MusCurte.drop(columns=['id_musica'])
dfUserA_MusNaoCurte = dfUserA_MusNaoCurte.drop(columns='id_musica')
#%%
print(len(df600kMusCurtem))
# %% Incluindo user A e user Abarrado no dataset User_MusCurtem
df600kMusCurtem = pd.concat([df600kMusCurtem, dfUserA_MusCurte, dfUserA_MusNaoCurte], ignore_index=True, verify_integrity=True)        

# caso rode esse componente duas vezes seguidas
df600kMusCurtem.drop_duplicates()

# %%
df600kMusCurtem.to_pickle('./FeatureStore/MusUsers.pickle')

# %%
print(len(df600kMusCurtem))
print(df600kMusCurtem.tail())
