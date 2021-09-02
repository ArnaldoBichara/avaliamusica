# vamos analisar como é essa base de user
# pergunta: qual a distribuição de músicas por user?

#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> Analisa Users')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  

#%%
dfMusUsers.describe()
#%%
#dfCountPerUser = dfMusUsers['userid'].value_counts().to_frame()
dfCountPerUser = dfMusUsers.copy()
dfCountPerUser['nrows']=1
dfCountPerUser = dfCountPerUser.groupby('userid')['nrows'].sum().reset_index()
#%% análise da distribuição de músicas por user 
dfCountPerUser.describe()
#%%
dfCountPerUser.head(50)
#%%
dfCountPerUser.max()
#%% 
dfCountPerUser.info
#%%
dfCountPerUser.index
#%% Criando filtro
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']>55]
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']<3194]
#%% lista de Users a filtrar
listaUsersAManter = list(dfCountPerUser['userid'])
#%%
print (listaUsersAManter)
#%%
dfMusUsers['userid']
#%%
dfMusUsers[dfMusUsers['userid'].isin(listaUsersAManter)]

# %%
#%%
dfCountPerUser.hist(bins=1000, figsize=(18,16))
plt.savefig("./Analises/Histograma Users.pdf")

# %%
