# Descobre os vizinhos do User A e do User Abarra

#%% Importando packages
from numpy.ma import sort
import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import confusion_matrix
import os

logging.basicConfig(filename='./Analises/preprocessamentoColab.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ColabProcessaUsuariosVizinhos: DescobreVizinhos de UserA e UserAbarra')

dfMusUsersColab = pd.read_pickle ("./FeatureStore/ColabMusUsersEsparsa.pickle")
listaDominioDeMusicas = pd.read_pickle ("./FeatureStore/DominioMusicasColab.pickle").to_list()

logging.info("MusUsersColab lido. shape %s", dfMusUsersColab.shape)

#%% obtendo  MusUserAColab
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


# removendo linhas que tenham algum NaN
dfMusUsersColab=dfMusUsersColab.dropna()

MusUserAColab      = MontalistaMusUserColab (listMusUserA)
MusUserAbarraColab = MontalistaMusUserColab (listMusUserAbarra)

# liberando memória
del listaDominioDeMusicas
del listMusUserA
del listMusUserAbarra

#
k = 30
serMusColab = dfMusUsersColab.drop(columns=['user'])

#
# achando k vizinhos mais próximos de user A

neigh = NearestNeighbors(n_neighbors=k, metric='jaccard')
neigh.fit(serMusColab)
distancias, indices = neigh.kneighbors([MusUserAColab])

for i in range (0, len(indices[0])):
        logging.info ("vizinho de A %s com dist %s", dfMusUsersColab.loc[indices[0][i],'user'], distancias[0][i])
        print ("vizinho de A", dfMusUsersColab.loc[indices[0][i],'user'], distancias[0][i])

#
# matriz de confusao comparando userA com primeiro e último vizinhos
logging.info ("Matriz de confusao vizinhos de A:")
for i in range (0, k):
        confusionMatrixVizinho = confusion_matrix(MusUserAColab, serMusColab.loc[indices[0][i]])
        logging.info ("%s", confusionMatrixVizinho)

# liberando memória
del MusUserAColab

#
# achando k vizinhos mais próximos de user Abarra
distancias, indices = neigh.kneighbors([MusUserAbarraColab])

for i in range (0, len(indices[0])):
        logging.info ("vizinho de Abarra %s com dist %s", dfMusUsersColab.loc[indices[0][i],'user'], distancias[0][i])
        print ("vizinho de Abarra", dfMusUsersColab.loc[indices[0][i],'user'], distancias[0][i])

# matriz de confusao comparando userAbarra com primeiro e último vizinhos
logging.info ("Matriz de confusao vizinhos de Abarra:")
for i in range (0, k):
        confusionMatrixVizinho = confusion_matrix(MusUserAbarraColab, serMusColab.loc[indices[0][i]])
        logging.info ("%s", confusionMatrixVizinho)

# liberando memória
del MusUserAbarraColab
del dfMusUsersColab
del serMusColab

logging.info('\n<< ColabProcessaUsuariosVizinhos')

# %%
