###################
# Este componente lê e trata as playlists Curto e Não Curto do Usuário A,
# obtendo diretamente do Spotify,
# e gera a lista de features das musicas das duas playlists
###################
#%%
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
#        print("%4d %s %s" % (i, playlist['uri'],  playlist['name']))
        if playlist['name']=='Curto':
            IdPlaylistCurto = playlist['id']
        if playlist['name']=='Não curto':
            IdPlaylistNaoCurto = playlist['id']
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None

# Obtendo as audio_features das listas de playlists
#
# obtem Id das Músicas da lista de items fornecida pelo Spotipy
def getIdsMusicas (playlistItems):
    listIdMusicas=[]
    for i, item in enumerate (playlistItems['items']):
        idMusica = item['track']['id']
        listIdMusicas.append(idMusica)
    return listIdMusicas

def getListMusFeatures(user,playlist_id_curto, playlist_id_NaoCurto):
    # incluindo músicas da playlist curto
    playlistItems = sp.user_playlist_tracks (user, playlist_id_curto)
    listMusFeatures = sp.audio_features (getIdsMusicas(playlistItems))
    while playlistItems['next']:
        playlistItems = sp.next(playlistItems)
        listMusFeatures.extend(sp.audio_features(getIdsMusicas(playlistItems)))
    
    # incluindo músicas da playlist não curto
    playlistItems = sp.user_playlist_tracks(user,playlist_id_NaoCurto)
    listMusFeatures.extend(sp.audio_features(getIdsMusicas(playlistItems)))
    while playlistItems['next']:
        playlistItems = sp.next(playlistItems)
        listMusFeatures.extend(sp.audio_features(getIdsMusicas(playlistItems)))
    
    return listMusFeatures

ListUserAMusFeatures = getListMusFeatures(userA, IdPlaylistCurto, IdPlaylistNaoCurto)

# filtrando item vazio 
ListUserAMusFeatures = [i for i in ListUserAMusFeatures if i is not None]

# passando para dataFrame
dfUserAMusFeatures = pd.DataFrame(ListUserAMusFeatures)

# renomeando algumas colunas

dfUserAMusFeatures.rename ( columns = {'id':'id_musica',
                                'name':'musica',
                                'duration_ms':'duração_ms',
                                'energy': 'energia',
                                'key':'chave',
                                'mode':'modo'},
                            inplace=True ) 

# id_musica passa a ser o index
#dfUserAMusFeatures.set_index('id_musica',verify_integrity=True, inplace=True)

# removendo algumas colunas que não nos interessam
dfUserAMusFeatures.drop(columns=['type','uri','track_href','analysis_url'],inplace=True)

# lendo dfMusicasUserACurteENaoCurte, que será usado para incluir artista e musica 
#
dfMusicasUserACurteENaoCurte =  pd.read_pickle ("./arquivos intermediarios/UserA_MusCurteENaoCurte.pickle")  

# incluir artista e musica em dfUserAMusFeatures
#%%
#dfUserAMusFeatures = dfUserAMusFeatures.join(dfMusicasUserACurteENaoCurte, how='left', on='id_musica')
dfUserAMusFeatures = dfUserAMusFeatures.merge(dfMusicasUserACurteENaoCurte, how='left', on='id_musica')
#%%
list(dfUserAMusFeatures.columns.values)
#%%
# reordenando colunas
dfUserAMusFeatures = dfUserAMusFeatures[
      [ 'id_musica',
        'artista',
        'musica',
        'duração_ms',
        'danceability',
        'energia',
        'chave',
        'loudness',
        'modo',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'time_signature',
        ]]
#%%
# salvar dataframe em .pickle
dfUserAMusFeatures.to_pickle ("./arquivos intermediarios/UserAMus&Features.pickle")

#%%
# algumas verificações
print ("\nUser A Mus&Features:\n")
print(dfUserAMusFeatures.tail())
print(dfMusicasUserACurteENaoCurte.tail())

# %%
