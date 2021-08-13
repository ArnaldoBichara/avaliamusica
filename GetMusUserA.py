###################
# Este componente lê e trata as playlists Curto e Não Curto do Usuário A,
# obtendo diretamente do Spotify.
###################

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
logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )

logging.info('GetMusUsrA >>')

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
# a partir d0 result de uma playlist, lista Ids das Musicas
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
                             columns=['id_musica', 'artista','musica'])
dfMusicasUserANaoCurte = pd.DataFrame(listMusicasUserANaoCurte,
                            columns=['id_musica', 'artista','musica'])

# removendo itens repetidos (diferentes interpretações de um artista\música)
logging.info ("MusUserACurte Inicial: %s", len(dfMusicasUserACurte.index))
dfMusicasUserACurte.drop_duplicates()
logging.info ("MusCurteUserA removidos duplicados: %s", len(dfMusicasUserACurte.index))

logging.info ("MusUserANaoCurte Inicial: %s", len(dfMusicasUserANaoCurte.index))
dfMusicasUserANaoCurte.drop_duplicates()    
logging.info ("MusUserANaoCurte removidos duplicados: %s", len(dfMusicasUserANaoCurte.index))

# incluindo userid 0001 para userA e 0000 para userAbarrado (usuário oposto a user A)
dfMusicasUserACurte['userid'] = '0001'
dfMusicasUserANaoCurte['userid'] = '0000' 

# reordenando colunas
dfMusicasUserACurte = dfMusicasUserACurte[
      [ 'userid',
        'artista',
        'musica',
        'id_musica'
        ]]
dfMusicasUserANaoCurte = dfMusicasUserANaoCurte[
      [ 'userid',
        'artista',
        'musica',
        'id_musica'
        ]]        

# salvando datasets
dfMusicasUserACurte.to_pickle ("./FeatureStore/MusUserACurte.pickle")
dfMusicasUserANaoCurte.to_pickle ("./FeatureStore/MusUserANaoCurte.pickle")                            

logging.info('GetMusUsrA <<')
