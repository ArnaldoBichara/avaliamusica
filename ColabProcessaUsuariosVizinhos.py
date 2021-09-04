# Descobre os vizinhos do User A e do User Abarra

#%% Importando packages
from numpy.ma import sort
import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import confusion_matrix
from pathlib import Path

logging.basicConfig(filename='./Analises/preprocessamentoColab.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ColabProcessaUsuariosVizinhos: DescobreVizinhos de UserA e UserAbarra')

dfMusUsersColab = pd.read_pickle ("./FeatureStore/ColabMusUsersEsparsa.pickle")
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


# removendo linhas que tenham algum NaN
dfMusUsersColab=dfMusUsersColab.dropna()

MusUserAColab      = MontalistaMusUserColab (listMusUserA)
MusUserAbarraColab = MontalistaMusUserColab (listMusUserAbarra)

# liberando memória
del listaDominioDeMusicas
del listMusUserA
del listMusUserAbarra

#
k = 20
serMusColab = dfMusUsersColab.drop(columns=['user'])

#
# achando k vizinhos mais próximos de user A

neigh = NearestNeighbors(n_neighbors=k, metric='sokalsneath')
neigh.fit(serMusColab)
distancias, indices = neigh.kneighbors([MusUserAColab])

#%% salvando vizinhos em arquivo VizinhosUserA.pickle
VizinhosUserA = Path("./FeatureStore/ColabVizinhosUserA.pickle")
if VizinhosUserA.is_file():
        dfVizinhosUserA = pd.read_pickle ("./FeatureStore/ColabVizinhosUserA.pickle")
else:
        dfVizinhosUserA = pd.DataFrame (columns=['userid',"distancia"])
for i in range (0, len(indices[0])):
        vizinho={'userid': dfMusUsersColab.loc[indices[0][i],'user'],
                 'distancia': distancias[0][i]}
        df= pd.DataFrame(data=vizinho, index=[i])
        dfVizinhosUserA = dfVizinhosUserA.append(df, ignore_index=True)
dfVizinhosUserA.to_pickle("./FeatureStore/ColabVizinhosUserA.pickle")

#%% liberando memória
del MusUserAColab

#%%
#
# achando k vizinhos mais próximos de user Abarra
distancias, indices = neigh.kneighbors([MusUserAbarraColab])


#%%
#%% salvando vizinhos em arquivo VizinhosUserAbarra.pickle
VizinhosUserAbarra = Path("./FeatureStore/ColabVizinhosUserAbarra.pickle")
if VizinhosUserAbarra.is_file():
        dfVizinhosUserAbarra = pd.read_pickle ("./FeatureStore/ColabVizinhosUserAbarra.pickle")
else:
        dfVizinhosUserAbarra = pd.DataFrame (columns=['userid',"distancia"])
for i in range (0, len(indices[0])):
        vizinho={'userid': dfMusUsersColab.loc[indices[0][i],'user'],
                 'distancia': distancias[0][i]}
        df= pd.DataFrame(data=vizinho, index=[i])
        dfVizinhosUserAbarra = dfVizinhosUserAbarra.append(df, ignore_index=True)
dfVizinhosUserAbarra.to_pickle("./FeatureStore/ColabVizinhosUserA.pickle")


# liberando memória
del MusUserAbarraColab
del dfMusUsersColab
del serMusColab

logging.info('\n<< ColabProcessaUsuariosVizinhos')

# %%
