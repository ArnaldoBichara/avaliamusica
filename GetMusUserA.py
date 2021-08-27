###################
# Este componente lê e trata as playlists Curto e Não Curto do Usuário A,
# obtendo diretamente do Spotify.
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import logging
from time import gmtime, strftime

# iniciando logging
logging.basicConfig(filename='./Analises/preprocessamento.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

logging.info('>> GetMusUsrA')

# conectando no spotify
scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

# User A <= 'jmwyg3knv7jv8tu84b19jxu3p'
userA = 'jmwyg3knv7jv8tu84b19jxu3p'

#  obtendo id das playlists Curto e Não Curto do user
#
playlists = sp.user_playlists(userA)

while playlists:
    for i, playlist in enumerate (playlists['items']):
        if playlist['name']=='Curto':
            IdPlaylistCurto = playlist['id']
        if playlist['name']=='Não curto':
            IdPlaylistNaoCurto = playlist['id']
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None

#
# a partir do result de uma playlist, lista Ids das Musicas
#
def getPartialMusicas (result):
    listMusicas=[]
    for i, item in enumerate (result['items']):
        arrayMusica = [item['track']['id'],item['track']['artists'][0]['name'],item['track']['name']]
        listMusicas.append(arrayMusica)
    return listMusicas

# obtendo músicas de uma playlist
def getMusicas(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    listMusicas = getPartialMusicas(results)
    while results['next']:
        results = sp.next(results)
        listMusicas.extend(getPartialMusicas(results))
    return listMusicas

# Obtendo lista de ids das playlists Curto e NaoCurto
listMusicasUserACurte = getMusicas(userA, IdPlaylistCurto)
listMusicasUserANaoCurte = getMusicas(userA, IdPlaylistNaoCurto)

dfMusicasUserACurte = pd.DataFrame(listMusicasUserACurte,
                            columns=['id_musica','artista','musica'])
dfMusicasUserANaoCurte = pd.DataFrame(listMusicasUserANaoCurte,
                            columns=['id_musica','artista','musica'])

# Criando coluna interpretacao e removendo colunas artista e musica
dfMusicasUserACurte['interpretacao']=dfMusicasUserACurte['artista'].str.upper()+":>"+dfMusicasUserACurte['musica'].str.upper()
del dfMusicasUserACurte['artista']
del dfMusicasUserACurte['musica']
dfMusicasUserANaoCurte['interpretacao']=dfMusicasUserANaoCurte['artista'].str.upper()+":>"+dfMusicasUserANaoCurte['musica'].str.upper()
del dfMusicasUserANaoCurte['artista']
del dfMusicasUserANaoCurte['musica']
#%%
#print (dfMusicasUserACurte[dfMusicasUserACurte['interpretacao'].str.contains("Beto Guedes:>Quando Te Vi", na= False, case=False)].to_string())

#%%
# removendo itens repetidos (diferentes interpretações de um artista\música)
logging.info ("MusUserACurte Inicial: %s", dfMusicasUserACurte.shape)
dfMusicasUserACurte.drop_duplicates(inplace=True, ignore_index=True)
logging.info ("MusCurteUserA removidos duplicados: %s", dfMusicasUserACurte.shape)

logging.info ("MusUserANaoCurte Inicial: %s", dfMusicasUserANaoCurte.shape)
dfMusicasUserANaoCurte.drop_duplicates(inplace=True, ignore_index=True)    
logging.info ("MusUserANaoCurte removidos duplicados: %s", dfMusicasUserANaoCurte.shape)

#%%
dfMusicasUserACurte.tail()
#%%
# salvando datasets
dfMusicasUserACurte.to_pickle ("./FeatureStore/MusUserACurte.pickle")
dfMusicasUserANaoCurte.to_pickle ("./FeatureStore/MusUserANaoCurte.pickle")                            

logging.info ("MusUserACurte.head %s", dfMusicasUserACurte.head())
logging.info('<< GetMusUsrA')

# %%
