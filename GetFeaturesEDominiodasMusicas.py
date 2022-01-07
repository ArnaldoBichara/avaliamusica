#################
# Este componente é responsável por ler o dataset spotify 600k tracks
# e montar dois datasets em pickle
# - AudioFeatures, com os atributos das músicas
# - DominioDasMusicas, com a lista de músicas
#################

#%%
# Importando packages
import pandas as pd
import numpy as np
import logging

logging.basicConfig(filename='./Analises/preprocessamento.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> GetFeaturesEDominio')

# lendo users dataset através de read_csv
dfSpotMusicas = pd.read_csv('./datasets/spotify600k_tracks.csv', 
                            nrows=108e6,
                            usecols = ['id', 'artists', 'name', 'duration_ms','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo', 'time_signature'],
                            #quoting = csv.QUOTE_NONE
                            )

logging.info ('GetFeaturesEDominio: spotify600k_tracks shape = %s', dfSpotMusicas.shape)

# Filtro: transforma lista de artists no artista principal (o primeiro da lista)
 
def obtem_primeiro_artista (data): # Clean the data because it are messy strings:
    new_data = []
    # para cada artists do df
    for i in data.values:
        # obtém primeiro artista da lista
        j = i[0].split(',')[0]
        # limpa string e inclui na lista
        new_data.append(j.replace('[', '').replace("'", '').replace(']', '').strip())
        # adiciona à lista
    return new_data

dfSpotMusicas['artista'] = obtem_primeiro_artista(dfSpotMusicas[['artists']])
dfSpotMusicas = dfSpotMusicas.drop(['artists'], axis=1)

# Criando coluna interpretacao e removendo colunas artista e musica
dfSpotMusicas['interpretacao']=dfSpotMusicas['artista'].str.upper()+":>"+dfSpotMusicas['name'].str.upper()

del dfSpotMusicas['artista']
del dfSpotMusicas['name']

dfSpotMusicas.rename(columns={'id':'id_musica'}, inplace=True)

dfSpotMusicas.drop_duplicates(subset='interpretacao', inplace=True, ignore_index=True)
# removendo linhas que tenham algo com NaN
dfSpotMusicas=dfSpotMusicas.dropna()

# Criando DominioDasMusicas
dominioMusicas = dfSpotMusicas[['interpretacao','id_musica']].copy()

# removendo algumas colunas não úteis para AudioFeatures
del dfSpotMusicas['interpretacao']

# incluindo músicas do UserA no dominio das Músicas
dfUserACurte    =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
dfUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  

dominioMusicas = pd.concat([dfUserACurte, dfUserANaoCurte, dominioMusicas], ignore_index=True, verify_integrity=True)        

logging.info ("Dominio antes de remover duplicates =%s", dominioMusicas.shape)
dominioMusicas.drop_duplicates(subset='interpretacao', inplace=True, ignore_index=True)
# removendo linhas que tenham algo com NaN
dominioMusicas=dominioMusicas.dropna()
# removendo linhas onde interpretacao não é uma string válida (por algum motivo isso está acontecendo)
dominioMusicas = dominioMusicas[dominioMusicas['interpretacao'].apply(lambda x: isinstance(x,str))]
logging.info ("Dominio apos remover duplicates =%s", dominioMusicas.shape)

dominioMusicas.reset_index(drop=True)

# ordenando dominioMusicas para tornar acesso mais rápido
dominioMusicas.sort_values(by=['interpretacao'], kind='quicksort', inplace=True)

logging.info ('GetFeaturesEDominio: dominio shape = %s', dominioMusicas.shape)
logging.info ('GetFeaturesEDominio: AudioFeatures shape = %s', dfSpotMusicas.shape)

dfSpotMusicas.to_pickle('./FeatureStore/AudioFeatures.pickle')
dominioMusicas.to_pickle('./FeatureStore/DominioDasMusicas.pickle')

logging.info('<< GetFeaturesEDominio')

# %%
