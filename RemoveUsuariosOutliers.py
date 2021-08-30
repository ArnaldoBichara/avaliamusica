# vamos analisar e filtrar base de user
# pergunta: qual a distribuição de músicas por user?

#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Analises/preprocessamento.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> RemoveUsuariosOutliers')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  


# dataframe para contar linhas de usuário
dfCountPerUser = dfMusUsers.copy()
dfCountPerUser['nrows']=1
dfCountPerUser = dfCountPerUser.groupby('userid')['nrows'].sum().reset_index()
#%%
# filtrando users dentro da faixa determinada
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']>200]
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']<800]

listaUsersAManter = list(dfCountPerUser['userid'])

# filtrando users definidos na lista de users
dfMusUsers= dfMusUsers[dfMusUsers['userid'].isin(listaUsersAManter)]
#%%
dfMusUsers.reset_index(inplace=True)

#%% removendo linhas que tenham algo com NaN
dfMusUsers.dropna(inplace=True)

#%%
dfCountPerUser.hist(bins=1000, figsize=(18,16))
plt.savefig("./Analises/Histograma Users.pdf")

#%%
logging.info ('MusUsers shape apos filtro %s', dfMusUsers.shape)
logging.info ('MusUsers describe apos filtro %s', dfMusUsers.describe())
logging.info ('dfCountPerUser describe apos filtro %s', dfCountPerUser.describe())


# salvando dataset
dfMusUsers.to_pickle ("./FeatureStore/MusUsers.pickle")

logging.info('<< RemoveUsuariosOutliers')

#

# %%
