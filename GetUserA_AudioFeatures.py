###################
# Este componente lê e trata as playlists Curto e Não Curto do Usuário A,
# obtendo diretamente do Spotify,
# e gera lista de features das musicas das duas playlists
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

def getListMusFeatures(user,playlist_id):
    # incluindo músicas da playlist
    playlistItems = sp.user_playlist_tracks (user, playlist_id)
    listMusFeatures = sp.audio_features (getIdsMusicas(playlistItems))
    while playlistItems['next']:
        playlistItems = sp.next(playlistItems)
        listMusFeatures.extend(sp.audio_features(getIdsMusicas(playlistItems)))
    
    return listMusFeatures
#%%
ListMusFeaturesUserA = getListMusFeatures(userA, IdPlaylistCurto)
ListMusFeaturesUserAbarra = getListMusFeatures(userA, IdPlaylistNaoCurto)
#%%
print(len(ListMusFeaturesUserA))
print(len(ListMusFeaturesUserAbarra))
#%%
# filtrando item vazio 
ListMusFeaturesUserA = [i for i in ListMusFeaturesUserA if i is not None]
ListMusFeaturesUserAbarra = [i for i in ListMusFeaturesUserAbarra if i is not None]

# passando para dataFrame
dfUserAMusFeatures = pd.DataFrame(ListMusFeaturesUserA)
dfUserAbarraMusFeatures = pd.DataFrame(ListMusFeaturesUserAbarra)


# renomeando algumas colunas

dfUserAMusFeatures.rename ( columns = {'id':'id_musica',
                                'name':'musica'},
                            inplace=True ) 


dfUserAbarraMusFeatures.rename ( columns = {'id':'id_musica',
                                'name':'musica'},
                                inplace=True ) 

# removendo algumas colunas que não nos interessam
dfUserAMusFeatures.drop(columns=['type','uri','track_href','analysis_url'],inplace=True)
dfUserAbarraMusFeatures.drop(columns=['type','uri','track_href','analysis_url'],inplace=True)
#%%
print (dfUserAMusFeatures.shape)
print (dfUserAbarraMusFeatures.shape)
#%%
# lendo dfMusicasUserACurteENaoCurte, que será usado para incluir artista e musica 
#
dfMusicasUserACurte =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
dfMusicasUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  

# incluir artista e musica em dfUserAMusFeatures
#%%
print(dfMusicasUserACurte.columns)
print(dfUserAMusFeatures.columns)
#%%
dfUserAMusFeatures = dfUserAMusFeatures.merge(dfMusicasUserACurte, how='left', on='id_musica')
dfUserAbarraMusFeatures = dfUserAbarraMusFeatures.merge(dfMusicasUserANaoCurte, how='left', on='id_musica')

#%%
list(dfUserAMusFeatures.columns.values)
#%%
# reordenando colunas
dfUserAMusFeatures = dfUserAMusFeatures[
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
dfUserAbarraMusFeatures = dfUserAbarraMusFeatures[
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

#%%
# salvar dataframe em .pickle
dfUserAMusFeatures.to_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")
dfUserAbarraMusFeatures.to_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")

#%%
# algumas verificações
print(dfUserAMusFeatures.tail())
print(dfUserAbarraMusFeatures.tail())
# %%
