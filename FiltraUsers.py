# vamos analisar e filtrar base de user
# pergunta: qual a distribuição de músicas por user?

#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> Filtra Users')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  

#%%
logging.info ('dfMusUsers describe %s', dfMusUsers.describe())
#%% dataframe para contar linhas de usuário
#dfCountPerUser = dfMusUsers['userid'].value_counts().to_frame()
dfCountPerUser = dfMusUsers.copy()
dfCountPerUser['nrows']=1
dfCountPerUser = dfCountPerUser.groupby('userid')['nrows'].sum().reset_index()
#%% análise da distribuição de músicas por user 
logging.info ('contagem de users %s', dfCountPerUser.describe())
#%% filtrando users dentro da faixa determinada
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']>55]
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']<3194]
listaUsersAManter = list(dfCountPerUser['userid'])
#%% filtrando users definidos na lista de users
dfMusUsers= dfMusUsers[dfMusUsers['userid'].isin(listaUsersAManter)]
#%%
dfCountPerUser.hist(bins=1000, figsize=(18,16))
plt.savefig("./Resultado das Análises/Histograma Users.pdf")

# salvando dataset
dfMusUsers.to_pickle ("./FeatureStore/MusUsers.pickle")

logging.info('<< Filtra Users')

# %%
