# vamos analisar e filtrar base de user
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
logging.info('\n>> ColabFiltraFaixaDeUsuarios')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsersNoDominio.pickle")  


# dataframe para contar linhas de usuário
dfCountPerUser = dfMusUsers.copy()
dfCountPerUser['nrows']=1
dfCountPerUser = dfCountPerUser.groupby('userid')['nrows'].sum().reset_index()
#
# filtrando users dentro da faixa determinada
minplay = 1501
maxplay = 5000
logging.info ('filtrando usuários com playlist entre %s e %s', minplay, maxplay)
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']>minplay]
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']<maxplay]

listaUsersAManter = list(dfCountPerUser['userid'])

# filtrando users definidos na lista de users
dfMusUsers= dfMusUsers[dfMusUsers['userid'].isin(listaUsersAManter)]
#
dfMusUsers = dfMusUsers.reset_index(level=0, drop=True)

# removendo linhas que tenham algo com NaN
dfMusUsers.dropna(inplace=True)

#
dfCountPerUser.hist(bins=1000, figsize=(18,16))
plt.savefig("./Analises/Histograma Users.pdf")

#
logging.info ('MusUsers shape apos filtro %s', dfMusUsers.shape)
logging.info ('MusUsers describe apos filtro %s', dfMusUsers.describe())
logging.info ('dfCountPerUser describe apos filtro %s', dfCountPerUser.describe())


# salvando dataset
dfMusUsers.to_pickle ("./FeatureStore/MusUsersFiltradas.pickle")

logging.info('\n<< ColabFiltraFaixaDeUsuarios')


# %%
