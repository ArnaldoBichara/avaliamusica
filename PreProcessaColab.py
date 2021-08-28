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
import bisect

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> PreProcessaColab')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  

#%%
dfMusUsers.head
#%%

dfMusUsers.reset_index(inplace=True, drop=True)
lstDominioDeMusicas = dfMusUsers['id_musica'].drop_duplicates().sort_values().to_list()
lstUserIds = dfMusUsers['userid'].drop_duplicates().to_list()
#%%
#print (lstDominioDeMusicas[0:10])
#%%
#print (dfMusUsers[dfMusUsers['id_musica']=='0'])
#%%
if os.path.isfile("./FeatureStore/MusUsersListaOrd.pickle"):
        dfMusUsersList = pd.read_pickle ("./FeatureStore/MusUsersListaOrd.pickle")
else:  
        dfMusUsersList = dfMusUsers.groupby (by='userid')['id_musica'].apply(lambda x: list(x.sort_values()) )
       
        dfMusUsersList.to_pickle ("./FeatureStore/MusUsersListaOrd.pickle")

#%%
#print (dfMusUsersList['00055176fea33f6e027cd3302289378b'])

#%%
#print (dfMusUsersList[lstUserIds[1]])

#%% A fazer: 
# rotina que monsta lista 0,1 para um user, por lista de musicas
# rowUserMusBin para cada User
def MontaRowUserMusBin (userid):
        listaMusUser = dfMusUsersList[userid]
        tamListaMusUser = len (listaMusUser)
        rowUserMusBin = [userid]
        for mus in lstDominioDeMusicas:
                # verifica se mus está na lista de músicas do User
                indice = np.searchsorted(listaMusUser, mus)
                if (indice != tamListaMusUser) and (mus==listaMusUser[indice]) :
                        rowUserMusBin.append(1)  # musica encontrada
                else :
                        rowUserMusBin.append(0)  # musica não encontrada          
        return rowUserMusBin

CONTINUAR DAQUI
row = MontaRowUserMusBin ('00055176fea33f6e027cd3302289378b')


#%% exemplo
lstDominioDeMusicas = ['00a','00b','00c','00d']
colunas=['user']+lstDominioDeMusicas

rowUserMusBin0=['User0', 0, 1, 1, 1]
rowUserMusBin1=['User1', 1, 1, 1, 1]
rowUserMusBin2=['User2', 0, 1, 0, 1]
rowUserMusBin3=['User3', 1, 0, 0, 0]
data = [rowUserMusBin0,
        rowUserMusBin1,
        rowUserMusBin2,
        rowUserMusBin3]
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
MusUsersColab = pd.DataFrame(lstUserIds, columns=['user'])

#%%
print (lstDominioDeMusicas[0])

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
dicionario.get(lstUserIds[0])
#%%

logging.info('<< PreProcessaColab')
