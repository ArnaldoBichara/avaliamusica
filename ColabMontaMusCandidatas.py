# A partir da totalidade de vizinhos por faixa,
# encontramos os N melhores vizinhos do UserA (e do UserAbarra)
# e descobrimos as músicas candidatas 
#   (o conjunto das músicas desses usuários que ainda não fazem parte do UserA)
# salvamos as músicas candidatas em .pickle

#%% Importando packages
import pandas as pd
import numpy as np
import logging
import pickle

# Número de vizinhos mais próximos a considerar
NVizinhos = 5

#
logging.basicConfig(filename='./Analises/preprocessamentoColab.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ColabMontaMusCandidatas')

VizinhosUserA       =  pd.read_pickle ("./FeatureStore/ColabVizinhosUserA.pickle")  
VizinhosUserAbarra  =  pd.read_pickle ("./FeatureStore/ColabVizinhosUserAbarra.pickle")  

# ordenando vizinhos por distancia e pegando apenas os N primeiros
VizinhosUserA.sort_values(by=['distancia'], inplace=True)
VizinhosUserAbarra.sort_values(by=['distancia'], inplace=True)

VizinhosUserA.reset_index(inplace=True, drop=True)
VizinhosUserAbarra.reset_index(inplace=True, drop=True)

# Pegando apenas primeiros N users mais próximos

VizinhosUserA      = VizinhosUserA[:NVizinhos]
logging.info ('\n%s Melhores vizinhos de UserA:', NVizinhos)
for i in range(NVizinhos):
    logging.info ("%s", VizinhosUserA.iloc[i]['userid'])

VizinhosUserAbarra = VizinhosUserAbarra[:NVizinhos]
logging.info ('\n%s Melhores vizinhos de UserAbarra:', NVizinhos)
for i in range(NVizinhos):
    logging.info ("%s", VizinhosUserAbarra.iloc[i]['userid'])

# Para cada user vizinho, 
#    encontra as músicas do User
#    'appenda' lista de músicas candidatas
#    'appenda' lista de músicas comuns com user A
# remove duplicates
# salva .pickle

MusUsers       =  pd.read_pickle ("./FeatureStore/MusUsersNoDominio.pickle")  

# 
# Montando músicas candidatas para UserA
#
listaMusCandUserA = []
for i in range(NVizinhos):
    vizinho = VizinhosUserA.iloc[i]['userid']
    MusUser = MusUsers[MusUsers['userid']==vizinho]['id_musica'].tolist()
    listaMusCandUserA.extend(MusUser)
    
# removendo duplicados na lista
listaMusCandUserA = list(set(listaMusCandUserA))

# removendo itens que já estão na playlist do UserA
MusUserA       =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")['id_musica'].tolist()

listaFinal = list(filter(lambda x: x not in MusUserA, listaMusCandUserA))

# Vamos salvar a lista interseccao apenas para análise
listaInterseccao = list(filter(lambda x: x in MusUserA, listaMusCandUserA))

logging.info ("Total de músicas candidatas para UserA: %s", len(listaFinal))

# salvando lista de músicas candidatas em .pickle
with open('./FeatureStore/MusCandUserACurte.pickle', 'wb') as arq:
    pickle.dump(listaFinal, arq)
with open('./FeatureStore/MusInterseccaoVizinhosComA.pickle', 'wb') as arq:
    pickle.dump(listaInterseccao, arq)

    
# 
# Montando músicas candidatas para UserAbarra
#
listaMusCandUserAbarra = []
for i in range(NVizinhos):
    vizinho = VizinhosUserAbarra.iloc[i]['userid']
    MusUser = MusUsers[MusUsers['userid']==vizinho]['id_musica'].tolist()
    listaMusCandUserAbarra.extend(MusUser)
    
# removendo duplicados na lista
listaMusCandUserAbarra = list(set(listaMusCandUserAbarra))

# removendo itens que já estão na playlist do UserAbarra
MusUserAbarra       =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")['id_musica'].tolist()

listaFinal = list(filter(lambda x: x not in MusUserAbarra, listaMusCandUserAbarra))
listaInterseccao = list(filter(lambda x: x in MusUserAbarra, listaMusCandUserAbarra))

logging.info ("Total de músicas candidatas para UserAbarra: %s", len(listaFinal))

# salvando lista de músicas candidatas em .pickle
with open('./FeatureStore/MusCandUserANaoCurte.pickle', 'wb') as arq:
    pickle.dump(listaFinal, arq)    
with open('./FeatureStore/MusInterseccaoVizinhosComAbarra.pickle', 'wb') as arq:
    pickle.dump(listaInterseccao, arq)

#%%
logging.info('\n<< ColabMontaMusCandidatas')

# %%
