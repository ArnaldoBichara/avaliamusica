
#################
# Este componente lê o dataset de playlists do Spotify 
# e faz uma limpeza, removendo uma coluna, renomeando outras e removendo musicas em branco 
# Se necessárias outras transformações/limpeza, fazer aqui.
# O resultado é o arquivo pickle com a lista de users, cada um com suas músicas prediletas
#################

# Importando packages
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import logging
from time import gmtime, strftime

# iniciando logging de métricas
logging.basicConfig(filename='./Resultado das Análises/preprocessamento.log', 
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
# lendo dataset
dfMusicasUser = pd.read_csv('./datasets/spotify_playlists_dataset.csv', 
                            sep=',', escapechar='\\',
                            nrows=1.1e9,
                            error_bad_lines=False)

# Limpeza: renomeando algumas colunas e removendo coluna playlistname
dfMusicasUser.columns = ['userid', 'artista', 'musica', 'playlistname']
del dfMusicasUser ['playlistname']

logging.info('MusUsers shape inicial: %s', dfMusicasUser.shape)

# Limpeza: removendo musicas em branco
if dfMusicasUser['musica'].isnull().sum():
    dfMusicasUser = dfMusicasUser.dropna(axis = 0, subset = ['musica'])

# Limpeza: removendo linhas duplicadas 
dfMusicasUser.drop_duplicates(inplace = True)
dfMusicasUser.reset_index(drop=True)

logging.info('MusUsers, removidas mus em branco e linhas duplicadas %s', dfMusicasUser.shape)

# salvando dataset
dfMusicasUser.to_pickle ("./FeatureStore/MusUsers.pickle")

