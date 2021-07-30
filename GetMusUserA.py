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

dfMusicasUserACurteENaoCurte =  dfMusicasUserACurte.append (dfMusicasUserANaoCurte, verify_integrity=True, ignore_index=True)
# não deveria haver duplicados, mas vamos remover se houver por erro de digitação
dfMusicasUserACurteENaoCurte.drop_duplicates()
#dfMusicasUserACurteENaoCurte.set_index('id_musica',verify_integrity=True, inplace=True)

# removendo colunas desnecessárias
dfMusicasUserACurte = dfMusicasUserACurte.drop(columns=['id_musica'])
dfMusicasUserANaoCurte = dfMusicasUserANaoCurte.drop(columns='id_musica')

# removendo itens repetidos (diferentes interpretações de um artista\música)
print (len(dfMusicasUserACurte.index))
dfMusicasUserACurte.drop_duplicates()
print (len(dfMusicasUserACurte.index))

print (len(dfMusicasUserANaoCurte.index))
dfMusicasUserANaoCurte.drop_duplicates()    
print (len(dfMusicasUserANaoCurte.index))

# incluindo userid 0001 para userA e 0000 para userAbarrado (usuário oposto a user A)
dfMusicasUserACurte['userid'] = '0001'
dfMusicasUserANaoCurte['userid'] = '0000' 

# reordenando colunas
dfMusicasUserACurte = dfMusicasUserACurte[
      [ 'userid',
        'artista',
        'musica',
        ]]
dfMusicasUserANaoCurte = dfMusicasUserANaoCurte[
      [ 'userid',
        'artista',
        'musica',
        ]]        

# salvando datasets
dfMusicasUserACurte.to_pickle ("./FeatureStore/MusCurteUserA.pickle")
dfMusicasUserANaoCurte.to_pickle ("./FeatureStore/MusNaoCurteUserA.pickle")                            
dfMusicasUserACurteENaoCurte.to_pickle ("./FeatureStore/MusicasUserA.pickle")                            

# algumas verificações
print ("\nUser A curte:\n")
print (len(dfMusicasUserACurte.index))
print (dfMusicasUserACurte.tail())

print ("\nUser A não curte:\n")
print (len(dfMusicasUserANaoCurte.index))
print (dfMusicasUserANaoCurte.tail())

