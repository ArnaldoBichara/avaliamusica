# Descobre os vizinhos do User A e do User Abarra

#%% Importando packages
from numpy.ma import sort
import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
import os

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> DescobreVizinhos: DescobreVizinhos de UserA e UserAbarra')

dfMusUsersColab = pd.read_pickle ("./FeatureStore/MusUsersColab.pickle")
listaDominioDeMusicas = pd.read_pickle ("./FeatureStore/DominioMusicasColab.pickle").to_list()

logging.info("MusUsersColab lido. shape %s", dfMusUsersColab.shape)

# obtendo  MusUserAColab
listMusUserA      =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")['id_musica'].sort_values().to_list()
listMusUserAbarra =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")['id_musica'].sort_values().to_list()

logging.info ("listMusUsrA %s"     , len (listMusUserA))
logging.info ("listMusUsrAbarra %s", len (listMusUserAbarra))

# rotina que monsta lista 0,1 para um user, por lista de musicas
def MontalistaMusUserColab (listaMusUser):
        tamListaDominio = len (listaDominioDeMusicas)
        resposta = [0]*len(listaDominioDeMusicas)
        #busca listaMusUser em Dominio de Musicas
        indicesNoDominio = np.searchsorted(listaDominioDeMusicas, listaMusUser)
        i=0
        for indice in indicesNoDominio:
                if (indice != tamListaDominio) and (listaDominioDeMusicas[indice]==listaMusUser[i]) :
                        resposta[indice] =1  # musica encontrada
                i=i+1          
        return resposta

MusUserAColab      = MontalistaMusUserColab (listMusUserA)
MusUserAbarraColab = MontalistaMusUserColab (listMusUserAbarra)

#%%
# achando k vizinhos mais próximos de user A
k = 10
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(dfMusUsersColab.drop(columns=['user']))
distancias, indices = neigh.kneighbors([MusUserAColab])

logging.info ("vizinhos de A dist:%s", distancias)

for row in indices:
        for indice in row:
                logging.info ("vizinho de A %s", dfMusUsersColab.loc[indice,'user'])

# achando k vizinhos mais próximos de user Abarra
distancias, indices = neigh.kneighbors([MusUserAbarraColab])

logging.info ("vizinhos de Abarra dist:%s", distancias)

for row in indices:
        for indice in row:
                logging.info ("vizinho de Abarra %s", dfMusUsersColab.loc[indice,'user'])

#%%

logging.info('<< DescobreVizinhos')
