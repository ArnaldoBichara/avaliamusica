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

logging.info ("ColabVizinhosUserA shape %s", VizinhosUserA.shape)
logging.info ("ColabVizinhosUserAbarra shape %s", VizinhosUserAbarra.shape)

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
DominioDasMusicas =  pd.read_pickle ("./FeatureStore/DominioDasMusicas.pickle")  
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

#passando músicas para dataframe e incluindo interpretacao
#%%
dfMusCandUserACurte = pd.DataFrame (listaFinal, columns=['id_musica'])
dfMusCandUserACurte['interpretacao']=''
for index, row in dfMusCandUserACurte.iterrows():
    print (index)
    dfMusCandUserACurte['interpretacao']= DominioDasMusicas[DominioDasMusicas['id_musica'] == row['id_musica']]['interpretacao']
logging.info ("MusCandUserACurte shape: %s", dfMusCandUserACurte.shape)
dfMusCandUserACurte.to_pickle('./FeatureStore/MusCandUserACurte.pickle')
#%%
#passando interseccao para dataframe e incluindo interpretacao
MusInterseccaoVizinhoscomA = pd.DataFrame (listaInterseccao, columns=['id_musica'])
MusInterseccaoVizinhoscomA['interpretacao']=''
for index, row in MusInterseccaoVizinhoscomA.iterrows():
    print (index)
    MusInterseccaoVizinhoscomA['interpretacao']= DominioDasMusicas[DominioDasMusicas['id_musica'] == row['id_musica']]['interpretacao']
logging.info ("MusInterseccaoVizinhoscomA shape: %s", MusInterseccaoVizinhoscomA.shape)
MusInterseccaoVizinhoscomA.to_pickle('./FeatureStore/MusInterseccaoVizinhoscomA.pickle')
#%%    
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

#passando músicas para dataframe e incluindo interpretacao
#%%
dfMusCandUserANaoCurte = pd.DataFrame (listaFinal, columns=['id_musica'])
dfMusCandUserANaoCurte['interpretacao']=''
for index, row in dfMusCandUserANaoCurte.iterrows():
    print (index)
    dfMusCandUserANaoCurte['interpretacao']= DominioDasMusicas[DominioDasMusicas['id_musica'] == row['id_musica']]['interpretacao']
logging.info ("MusCandUserANaoCurte shape: %s", dfMusCandUserANaoCurte.shape)
dfMusCandUserANaoCurte.to_pickle('./FeatureStore/MusCandUserACurte.pickle')

logging.info ("lista de Interseccao UserAbarra: %s", len(listaInterseccao))

#%%
logging.info('\n<< ColabMontaMusCandidatas')

# %%
