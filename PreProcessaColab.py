# este componente prepara o dataset MusUserColab usado pelo
# algoritmo de Colaboração de Users para descobrir quais os vizinhos
# do UserA e do UserAbarra


#%% Importando packages
from numpy.ma import sort
import pandas as pd
import numpy as np
import seaborn as sns
import logging
from sklearn.neighbors import NearestNeighbors
import os

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> PreProcessaColab')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  

dfMusUsers.reset_index(inplace=True)

listademusicas = dfMusUsers['id_musica'].drop_duplicates().to_list()
listadeusers = dfMusUsers['userid'].drop_duplicates().to_list()

#%%
if os.path.isfile("./FeatureStore/MusUsersListaOrd.pickle"):
  dfMusUsersList = pd.read_pickle ("./FeatureStore/MusUsersListaOrd.pickle")
else:  
        def ordenaELista(s):
                return list(s.sort_values())
        dfMusUsersList = dfMusUsers.groupby (by='userid')['interpretacao'].apply(lambda x: ordenaELista(x))
       
        dfMusUsersList.to_pickle ("./FeatureStore/MusUsersListaOrd.pickle")

#%%
print (dfMusUsersList['00055176fea33f6e027cd3302289378b'])

#%%
print (dfMusUsersList[listadeusers[1]])
#%% montagem exemplo
listademusicas = ['00a','00b','00c','00d']
colunas=['user']+listademusicas

rowUser0=['User0', 0, 1, 1, 1]
rowUser1=['User1', 1, 1, 1, 1]
rowUser2=['User2', 0, 1, 0, 1]
rowUser3=['User3', 1, 0, 0, 0]
data = [rowUser0,
        rowUser1,
        rowUser2,
        rowUser3]
MusUsersColab = pd.DataFrame(data, columns=colunas)
print (MusUsersColab)

#%% calcula vizinhos do UserA
k = 3
musUserA=[ 1, 1, 1, 1]
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(MusUsersColab.drop(columns=['user']))
distancias, indices = neigh.kneighbors([musUserA])
for row in indices:
        for indice in row:
                print (MusUsersColab.loc[indice,'user'])

#%%

#%%
MusUsersColab = pd.DataFrame(listadeusers, columns=['user'])

#%%
print (listademusicas[0])

#%% descobrindo se user x possui música y na playlist
dfMusUsers[['userid','interpretacao']]

#%% agrupando músicas por userid
#%% removendo linhas que tenham algo com NaN
print(dfMusUsers.shape)
dfMusUsers.dropna(inplace=True)
print(dfMusUsers.shape)

#%%
print (dfMusUsers[dfMusUsers['userid']=='00055176fea33f6e027cd3302289378b']['interpretacao'].head())
#%%
#%%
#dfMusUserList = dfMusUsers.groupby (by='userid')['interpretacao'].apply(list).apply(lambda x: x.sort()).reset_index(name='lista')

#%%
print (dfMusUserList.head())

#%%
print (dfMusUserList['00055176fea33f6e027cd3302289378b'].head())
#%%
print (dfMusUserList['00055176fea33f6e027cd3302289378b'])
#%%
print (dfMusUserList.iloc[0]['lista'])
#%% ao invés de usar groupby list, vamos ver como é com tupple
grouped = dfMusUsers.groupby (by='userid')['interpretacao']
df = grouped.aggregate(lambda x: tuple(x))
#%%
df.iloc[0]
#%% achando uma música na lista 
#%%
dicionario = dfUsuarios.groups
print(dicionario)
#%%
dicionario.get(listadeusers[0])
#%%

logging.info('<< PreProcessaColab')
