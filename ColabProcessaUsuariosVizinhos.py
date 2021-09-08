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

logging.basicConfig(filename='./Analises/processamentoColab.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ColabProcessaUsuariosVizinhos: DescobreVizinhos de UserA e UserAbarra')

dfMusUsersColab = pd.read_pickle ("./FeatureStore/ColabMusUsersEsparsa.pickle")
logging.info("ColabMusUsersEsparsa shape %s", dfMusUsersColab.shape)

# obtendo  matriz esparsa de User A
MusUserAColab      =  pd.read_pickle ("./FeatureStore/ColabMusUserAEsparsa.pickle")
MusUserAbarraColab =  pd.read_pickle ("./FeatureStore/ColabMusUserAbarraEsparsa.pickle")



k = 20
metrica ='jaccard'
#metrica = 'sokalsneath')
serMusColab = dfMusUsersColab.drop(columns=['user'])

# Prepara algoritimo NearestNeighbors com Matriz esparsa
neigh = NearestNeighbors(n_neighbors=k, metric=metrica)
neigh.fit(serMusColab)

# achando k vizinhos mais próximos de user A e atualizando ColabVizinhosUserA.pickle com novos vizinhos
distancias, indices = neigh.kneighbors([MusUserAColab])

VizinhosUserA = Path("./FeatureStore/ColabVizinhosUserA.pickle")
if VizinhosUserA.is_file():
        dfVizinhosUserA = pd.read_pickle ("./FeatureStore/ColabVizinhosUserA.pickle")
else:
        dfVizinhosUserA = pd.DataFrame (columns=['userid',"distancia"]) # dataframe vazio

logging.info ("vizinhos de A:")
for i in range (0, len(indices[0])):
        indice = indices[0][i]
        vizinho={'userid': dfMusUsersColab.loc[indice,'user'],
                'distancia': distancias[0][i]}
        listaMusUser = dfMusUsersColab.loc[indice, dfMusUsersColab.columns !='user'].tolist()
        matriz_confusao = confusion_matrix (MusUserAColab, listaMusUser)
        logging.info ("%s %s", dfMusUsersColab.loc[indice,'user'], distancias[0][i])
        logging.info ("%s", matriz_confusao)              
        df= pd.DataFrame(data=vizinho, index=[i])
        dfVizinhosUserA = dfVizinhosUserA.append(df, ignore_index=True)
dfVizinhosUserA.to_pickle("./FeatureStore/ColabVizinhosUserA.pickle")
logging.info("vizinhos de A shape:%s", dfVizinhosUserA.shape)
#%% liberando memória
del MusUserAColab
del serMusColab

# achando k vizinhos mais próximos de user Abarra e atualizando ColabVizinhosUserAbarra.pickle com novos vizinhos
distancias, indices = neigh.kneighbors([MusUserAbarraColab])

VizinhosUserAbarra = Path("./FeatureStore/ColabVizinhosUserAbarra.pickle")
if VizinhosUserAbarra.is_file():
        dfVizinhosUserAbarra = pd.read_pickle ("./FeatureStore/ColabVizinhosUserAbarra.pickle")
else:
        dfVizinhosUserAbarra = pd.DataFrame (columns=['userid',"distancia"]) # dataframe vazio

logging.info ("vizinhos de Abarra:")
for i in range (0, len(indices[0])):
        indice = indices[0][i]
        vizinho={'userid': dfMusUsersColab.loc[indice,'user'],
                'distancia': distancias[0][i]}
        listaMusUser = dfMusUsersColab.loc[indice, dfMusUsersColab.columns !='user'].tolist()
        matriz_confusao = confusion_matrix (MusUserAbarraColab, listaMusUser)
        logging.info ("%s %s", dfMusUsersColab.loc[indice,'user'], distancias[0][i])
        logging.info ("%s", matriz_confusao)              
        df= pd.DataFrame(data=vizinho, index=[i])
        dfVizinhosUserAbarra = dfVizinhosUserAbarra.append(df, ignore_index=True)
        
dfVizinhosUserAbarra.to_pickle("./FeatureStore/ColabVizinhosUserAbarra.pickle")
logging.info("vizinhos de Abarra shape:%s", dfVizinhosUserAbarra.shape)
# liberando memória
del MusUserAbarraColab
del dfMusUsersColab

logging.info('\n<< ColabProcessaUsuariosVizinhos')

# %%
