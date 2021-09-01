# este componente prepara o dataset listaMusUserColab usado pelo
# algoritmo de Colaboração de Users para descobrir quais os vizinhos
# do UserA e do UserAbarra

#%% Importando packages
from numpy.ma import sort
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> PreProcessaColab')

#
dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
dfMusUsers.reset_index(inplace=True, drop=True)
listaUserIds = dfMusUsers['userid'].drop_duplicates().to_list()
listaDominioDeMusicas = pd.read_pickle ("./FeatureStore/DominioMusicasColab.pickle").to_list()

logging.info("listaDominioDeMusicas %s", len(listaDominioDeMusicas))
logging.info('listaUsers %s', len(listaUserIds))
#
#
# rotina que monsta lista 0,1 para um user, por lista de musicas
# rowUserMusBin para cada User
def MontaRowlistaMusUserColab (userid):
        listaMusUser = dfMusUsersList[userid]
        tamListaDominio = len (listaDominioDeMusicas)
        resposta = [0]*len(listaDominioDeMusicas)
        #busca listaMusUser em Dominio de Musicas
        indicesNoDominio = np.searchsorted(listaDominioDeMusicas, listaMusUser)
        i=0
        for indice in indicesNoDominio:
                if (indice != tamListaDominio) and (listaDominioDeMusicas[indice]==listaMusUser[i]) :
                        resposta[indice] =1  # musica encontrada
                i=i+1          
        return [userid]+resposta


if os.path.isfile("./FeatureStore/MusUsersColab.pickle")==False:
        dfMusUsersList = dfMusUsers.groupby (by='userid')['id_musica'].apply(lambda x: tuple(x.sort_values()) )

# definindo as colunas
        colunas=['user']
        colunas.extend (listaDominioDeMusicas)

# Monta lista de listas listaMusUserColab, user a user
        listaMusUserColab =[]
        i=0
        for user in listaUserIds:
                i=i+1;
                print (i)
                listaMusUserColab.append ( MontaRowlistaMusUserColab (user) )

# liberando memória
        del dfMusUsers
        del listaDominioDeMusicas
        del listaUserIds
        del dfMusUsersList
        
# montando dataframe e salvando em .pickle     
        dfMusUsersColab = pd.DataFrame (listaMusUserColab, columns=colunas)
        dfMusUsersColab.to_pickle ("./FeatureStore/MusUsersColab.pickle")

logging.info('<< PreProcessaColab')
