
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import csv


dfSpotMusicas= pd.read_pickle("./SpotifyMusicas.pickle")

dfArtistas = dfSpotMusicas[['artists']]
dfSpotMusicas[['artista']]=''
# First we have to take only the collaboration songs 
def obtem_musicas_em_colaboracao (data): # Takes only the artists with multiple artist
    new_data = []
    for i in data.to_numpy():
        if len(i[0].split(',')) > 1:
            new_data.append(i)
    return new_data

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

#artistas = obtem_musicas_em_colaboracao(dfArtistas)
dfSpotMusicas['artista'] = obtem_primeiro_artista(dfSpotMusicas[['artists']])
dfSpotMusicas = dfSpotMusicas.drop(['artists'], axis=1)

print (dfSpotMusicas.tail())




