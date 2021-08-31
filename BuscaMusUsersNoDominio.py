# Filtrando em MusUsers apenas aquelas que estejam no domínio
# Incluindo id_musica

#%% Importando packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Analises/preprocessamento.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> FiltraMusUsersNoDominio')


dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
#%%
dfDominioMusicas = pd.read_pickle ("./FeatureStore/DominioDasMusicas.pickle")

logging.info('MusUsers, antes do filtro %s', dfMusUsers.shape)

def BuscaInterpretacaoNoDominio (interpretacao):
    try:
        if (isinstance(interpretacao, str) == False):
            return '0'
        index = dfDominioMusicas['interpretacao'].values.searchsorted(interpretacao)
        if (interpretacao != dfDominioMusicas.iloc[index]['interpretacao']):
            return '0'
        else:
            return dfDominioMusicas.iloc[index]['id_musica']
    except IndexError:
        print ('Indice fora dos limites', index)    
        logging.info('Indice fora dos limites %s', index)
        return '0'

#%%
#print (dfMusUsers.head(100))       
#%%
if 'id_musica' not in dfMusUsers:
    dfMusUsers = dfMusUsers.assign(id_musica='0')
if 'ja_verificado' not in dfMusUsers:
    dfMusUsers = dfMusUsers.assign(ja_verificado=False)
#newmususers = pd.DataFrame(columns = ['userid','interpretacao', 'id_musica'])
itens_a_remover=[]
contador=0
for index, row in dfMusUsers.iterrows():
    contador=contador+1
    if (row['ja_verificado']==True):
        continue
    dfMusUsers.at[index,'ja_verificado']=True
    if (row['id_musica']=='0'):
        id_musica = BuscaInterpretacaoNoDominio (row['interpretacao'])
        if (contador%500000==0):
#            print (contador)
            dfMusUsers.drop(itens_a_remover, inplace=True)
            itens_a_remover=[]
            dfMusUsers.to_pickle ("./FeatureStore/MusUsers.pickle")

        if id_musica=='0':
            itens_a_remover.append(index)
            continue
        else:
            dfMusUsers.at[index,'id_musica']=id_musica
dfMusUsers.drop(itens_a_remover, inplace=True)

#%%
dfMusUsers.shape
#%%
#dfMusUsers[dfMusUsers['id_musica']=='0'].index
#dfMusUsers.drop (dfMusUsers[dfMusUsers['id_musica']=='0'].index, inplace=True)   
#dfMusUsers.to_pickle ("./FeatureStore/MusUsers.pickle")
    
#%% removendo coluna temporária
dfMusUsers.drop(columns=['ja_verificado'], inplace=True)
#
logging.info('MusUsers, depois do filtro %s', dfMusUsers.shape)

# salvando dataset
dfMusUsers.to_pickle ("./FeatureStore/MusUsers.pickle")
logging.info('<< FiltraMusUsersNoDominio')

#

# %%
