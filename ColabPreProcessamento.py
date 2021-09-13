# este componente prepara o dataset listaMusUserColab usado pelo
# algoritmo de Colaboração de Users para descobrir quais os vizinhos
# do UserA e do UserAbarra

#%% Importando packages
from numpy.ma import sort
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import sys

# Argumentos de entrada 
# minplay
# maxplay

if (len(sys.argv)<2):
  print ('argumentos obrigatorios: minplay e maxplay')
  quit()
faixa_minima = int(sys.argv[1])
faixa_maxima = int(sys.argv[2])
#%%
logging.basicConfig(filename='./Analises/processamentoColab.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ColabPreProcessamento')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsersFiltradas.pickle") 

###
# Fase de Filtragem de usuários dentro da faixa
###

# dataframe para contar linhas de usuário
dfCountPerUser = dfMusUsers.copy()
dfCountPerUser['nrows']=1
dfCountPerUser = dfCountPerUser.groupby('userid')['nrows'].sum().reset_index()

# filtrando users dentro da faixa determinada
logging.info ('filtrando users com playlist entre %s e %s', faixa_minima, faixa_maxima)
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']>faixa_minima]
dfCountPerUser = dfCountPerUser[dfCountPerUser['nrows']<faixa_maxima]

listaUsersAManter = list(dfCountPerUser['userid'])

# filtrando músicas apenas dos users definidos na lista de users
dfMusUsers = dfMusUsers[dfMusUsers['userid'].isin(listaUsersAManter)]
dfMusUsers = dfMusUsers.reset_index(level=0, drop=True)

###
# Fase de montagem da matrix esparsa Users x Dpmínio das Músicas 
###
 
listaUserIds = dfMusUsers['userid'].drop_duplicates().to_list()
listaDominioDeMusicas = pd.read_pickle ("./FeatureStore/DominioMusicasColab.pickle").to_list()

logging.info('listaUsers %s', len(listaUserIds))

# rotina que monsta lista True, False da lista de músicas para um dado user
def MontalistaMusUserColabPorLista (listaMusUser):
        tamListaDominio = len (listaDominioDeMusicas)
        resposta = [False]*len(listaDominioDeMusicas)
        #busca listaMusUser em Dominio de Musicas
        indicesNoDominio = np.searchsorted(listaDominioDeMusicas, listaMusUser)
        i=0
        for indice in indicesNoDominio:
                if (indice != tamListaDominio) and (listaDominioDeMusicas[indice]==listaMusUser[i]) :
                        resposta[indice] =True  # musica encontrada
                i=i+1          
        return resposta
#
# rotina que monsta lista True, False para um user, por userid
def MontaRowlistaMusUserColab (userid):
        listaMusUser = dfMusUsersList[userid]
        tamListaDominio = len (listaDominioDeMusicas)
        resposta = [False]*len(listaDominioDeMusicas)
        #busca listaMusUser em Dominio de Musicas
        indicesNoDominio = np.searchsorted(listaDominioDeMusicas, listaMusUser)
        i=0
        for indice in indicesNoDominio:
                if (indice != tamListaDominio) and (listaDominioDeMusicas[indice]==listaMusUser[i]) :
                        resposta[indice] =True  # musica encontrada
                i=i+1          
        return [userid]+resposta

# Se ainda não existe, monta matrix esparsa de user A e Abarra 
EsparsaUserA = Path("./FeatureStore/ColabMusUserAEsparsa.pickle")
if EsparsaUserA.is_file() == False:
        listMusUserA      =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")['id_musica'].sort_values().to_list()
        listMusUserAbarra =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")['id_musica'].sort_values().to_list()

        logging.info ("listMusUsrA %s"     , len (listMusUserA))
        logging.info ("listMusUsrAbarra %s", len (listMusUserAbarra))

        listaMusUserAColab      = MontalistaMusUserColabPorLista (listMusUserA)
        listaMusUserAbarraColab = MontalistaMusUserColabPorLista (listMusUserAbarra)

        with open('./FeatureStore/ColabMusUserAEsparsa.pickle', 'wb') as arq:
           pickle.dump(listaMusUserAColab, arq)

        with open('./FeatureStore/ColabMusUserAbarraEsparsa.pickle', 'wb') as arq:
           pickle.dump(listaMusUserAbarraColab, arq)

        #liberando memória
        del listMusUserA
        del listMusUserAbarra
        del listaMusUserAColab
        del listaMusUserAbarraColab

#%%
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
dfMusUsersColab.to_pickle ("./FeatureStore/ColabMusUsersEsparsa.pickle")

logging.info('\n<< ColabPreProcessamento')

# %%
