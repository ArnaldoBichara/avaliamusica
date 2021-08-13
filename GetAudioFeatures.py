#################
# Este componente é responsável por ler o dataset spotify 600k traks
# e montar um dataset com as músicas, apenas o artista principal (o primeiro)
# Se necessárias outras transformações/limpeza, fazer aqui.
# O resultado é o arquivo pickle com o dataset 
# além de uma lista das interpretações (artista+música) em excel
#################

#%%
# Importando packages
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import csv
import logging
from time import gmtime, strftime

logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('GetAudioFeatures >>')

# lendo users dataset através de read_csv
dfSpotMusicas = pd.read_csv('./datasets/spotify600k_tracks.csv', 
                            nrows=108e6,
                            usecols = ['id','artists','name','duration_ms','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo', 'time_signature'],
                            #quoting = csv.QUOTE_NONE
                            )

logging.info ('GetAudioFeatures: spotify600k_tracks shape = %s', dfSpotMusicas.shape)

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

# setando 'id' como index
#dfSpotMusicas.set_index('id_musica',verify_integrity=True, inplace=True);

dfSpotMusicas.rename(columns = {'id':'id_musica',
                                'name':'musica',
                                },
                     inplace=True)   
#%%
list(dfSpotMusicas.columns.values)
#%%
# reordenando colunas
dfSpotMusicas = dfSpotMusicas[
      [ 'id_musica',
        'artista',
        'musica',
        'duration_ms',
        'danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'time_signature',
        ]]
# salvando dataset
logging.info ('GetAudioFeatures: AudioFeatures shape = %s', dfSpotMusicas.shape)

dfSpotMusicas.to_pickle('./FeatureStore/AudioFeatures.pickle')



logging.info('GetAudioFeatures <<')
