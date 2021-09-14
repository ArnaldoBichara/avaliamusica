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
#NVizinhos = 7
NVizinhos = 3

#
logging.basicConfig(filename='./Analises/processamentoColab.log', 
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
#%%
# Pegando apenas primeiros N users mais próximos

VizinhosUserA      = VizinhosUserA[:NVizinhos]
VizinhosUserA.reset_index(inplace=True, drop=True)
logging.info ('\n%s Melhores vizinhos de UserA:', NVizinhos)
for i in range(NVizinhos):
    logging.info ("%s %s", VizinhosUserA.iloc[i]['userid'], VizinhosUserA.iloc[i]['distancia'])

VizinhosUserAbarra = VizinhosUserAbarra[:NVizinhos]
VizinhosUserAbarra.reset_index(inplace=True, drop=True)

logging.info ('\n%s Melhores vizinhos de UserAbarra:', NVizinhos)
for i in range(NVizinhos):
    logging.info ("%s %s", VizinhosUserAbarra.iloc[i]['userid'], VizinhosUserAbarra.iloc[i]['distancia'])

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

# ordenando dominioMusicas para tornar acesso mais rápido por id_musica
DominioDasMusicas.sort_values(by=['id_musica'], kind='quicksort', inplace=True)

def BuscaInterpretacaoNoDominio (id_musica):
    index = DominioDasMusicas['id_musica'].values.searchsorted(id_musica)
    return DominioDasMusicas.iloc[index]['interpretacao']

#passando músicas para dataframe e incluindo interpretacao
#%%
dfMusCandUserACurte = pd.DataFrame (listaFinal, columns=['id_musica'])
dfMusCandUserACurte['interpretacao']=''
for index, row in dfMusCandUserACurte.iterrows():
    print (index)
    dfMusCandUserACurte.iloc[index]['interpretacao']= BuscaInterpretacaoNoDominio(row['id_musica'])

#%%
logging.info ("MusCandUserACurte shape: %s", dfMusCandUserACurte.shape)
dfMusCandUserACurte.to_pickle('./FeatureStore/MusCandidatasCurte.pickle')
#%%
#passando interseccao para dataframe e incluindo interpretacao
MusInterseccaoVizinhoscomA = pd.DataFrame (listaInterseccao, columns=['id_musica'])
MusInterseccaoVizinhoscomA['interpretacao']=''
for index, row in MusInterseccaoVizinhoscomA.iterrows():
    print (index)
    MusInterseccaoVizinhoscomA.iloc[index]['interpretacao']= BuscaInterpretacaoNoDominio(row['id_musica'])
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
dfMusCandUserANaoCurte = pd.DataFrame (listaFinal, columns=['id_musica'])
dfMusCandUserANaoCurte['interpretacao']=''
for index, row in dfMusCandUserANaoCurte.iterrows():
    print (index)
    dfMusCandUserANaoCurte.iloc[index]['interpretacao']= BuscaInterpretacaoNoDominio(row['id_musica'])
logging.info ("MusCandUserANaoCurte shape: %s", dfMusCandUserANaoCurte.shape)
dfMusCandUserANaoCurte.to_pickle('./FeatureStore/MusCandidatasNaoCurte.pickle')

logging.info ("lista de Interseccao UserAbarra: %s", len(listaInterseccao))

#%%
logging.info('\n<< ColabMontaMusCandidatas')

# %%
