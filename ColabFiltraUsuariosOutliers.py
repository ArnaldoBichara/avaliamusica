# Para diminuir o escopo no Domínio das músicas, em função de recursos de memória, 
# vamos remover usuários outliers.
# Em outras palavras, a partir da análise da distribuição número de músicas por Usuário, 
# removemos usuários que tenham muitas músicas ou poucas músicas (de acordo com critério definido pela análise). 
# Apenas as músicas que estão nas playlists dos usuários, previamente filtrados ou de UserA 
# serão levadas em conta no domínio das músicas para análise colaborativa
# Também faz ordenação das músicas

#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging


#%%
logging.basicConfig(filename='./Analises/processamentoColab.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ColabFiltraUsuariosOutliers')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsersNoDominio.pickle")  

# dataframe para contar linhas de usuário
dfCountPerUser = dfMusUsers.copy()
dfCountPerUser['nrows']=1
dfCountPerUser = dfCountPerUser.groupby('userid')['nrows'].sum().reset_index()
#
# filtrando users dentro da faixa determinada
faixa_minima=50
faixa_maxima=2000
logging.info ('filtrando users com playlist entre %s e %s', faixa_minima, faixa_maxima)
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']>faixa_minima]
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']<faixa_maxima]

listaUsersAManter = list(dfCountPerUser['userid'])

# filtrando músicas apenas dos users definidos na lista de users
dfMusUsers= dfMusUsers[dfMusUsers['userid'].isin(listaUsersAManter)]
dfMusUsers = dfMusUsers.reset_index(level=0, drop=True)

logging.info ('MusUsers shape apos filtro %s', dfMusUsers.shape)
logging.info ('MusUsers describe apos filtro %s', dfMusUsers.describe())
logging.info ('dfCountPerUser describe apos filtro %s', dfCountPerUser.describe())

# salvando dataset MusUsers
dfMusUsers.to_pickle ("./FeatureStore/MusUsersFiltradas.pickle")

#
# Definindo domínio das músicas
#

# vamos incluir as músicas do User A
musUserA        =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
musUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  

domMusColab = pd.concat([musUserA[['id_musica', 'interpretacao']], musUserANaoCurte[['id_musica', 'interpretacao']],dfMusUsers[['id_musica', 'interpretacao']]], ignore_index=True, verify_integrity=True)        

# Removendo músicas que tenham nome (interpretacao) igual
domMusColab.drop_duplicates(subset='interpretacao', inplace=True)

# agora que removi as interpretacoes duplicadas, posso remover a coluna interpretacao
domMusColab = domMusColab.drop(columns=['interpretacao'])

# removendo itens que, por algum motivo, tenham id_musica igual
domMusColab.drop_duplicates(inplace=True)

# removendo linhas que tenham algo com NaN e removendo o index
domMusColab=domMusColab.dropna()
domMusColab.reset_index(drop=True, inplace=True)

# transformand df em series
domMusColab = domMusColab['id_musica']

# removendo linhas onde id_musica não é uma string válida (por algum motivo isso está acontecendo)
domMusColab= domMusColab[domMusColab.apply(lambda x: isinstance(x,str))]

# ordenando dominioMusicas para tornar acesso mais rápido
domMusColab.sort_values(kind='quicksort', inplace=True)

#
logging.info ("Dominio de musicas para Colab =%s", domMusColab.shape)
# salvando dataset
domMusColab.to_pickle ("./FeatureStore/DominioMusicasColab.pickle")

# removendo linhas que tenham algo com NaN
dfMusUsers.dropna(inplace=True)

#
dfCountPerUser.hist(bins=1000, figsize=(18,16))
plt.savefig("./Analises/Histograma Users.pdf")

logging.info('\n<< ColabFiltraUsuariosOutliers')

