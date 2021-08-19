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
import logging
from time import gmtime, strftime

logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('GetUserA_AudioFeatures >>')

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

ListMusFeaturesUserA = getListMusFeatures(userA, IdPlaylistCurto)
ListMusFeaturesUserAbarra = getListMusFeatures(userA, IdPlaylistNaoCurto)

# filtrando item vazio 
ListMusFeaturesUserA = [i for i in ListMusFeaturesUserA if i is not None]
ListMusFeaturesUserAbarra = [i for i in ListMusFeaturesUserAbarra if i is not None]

# passando para dataFrame
dfUserAMusFeatures = pd.DataFrame(ListMusFeaturesUserA)
logging.info ("UserAMusFeatures shape %s", dfUserAMusFeatures.shape)

dfUserAbarraMusFeatures = pd.DataFrame(ListMusFeaturesUserAbarra)
logging.info ("UserAMusbarraFeatures shape %s", dfUserAbarraMusFeatures.shape)

#%% Incluindo classe (0 - não curte / 1 - curte) e juntando dataframes
dfUserAMusFeatures['classe']= 1;
dfUserAbarraMusFeatures['classe'] = 0;
#%%
print (dfUserAMusFeatures.head())
#%%
print (dfUserAbarraMusFeatures.head())
#%%
dfUserAMusFeatures = pd.concat([dfUserAMusFeatures, dfUserAbarraMusFeatures], ignore_index=True, verify_integrity=True)        

# removendo algumas colunas que não nos interessam
dfUserAMusFeatures.drop(columns=['id','type','uri','track_href','analysis_url'],inplace=True)
#%%
logging.info ("UserAMusFeatures shape %s", dfUserAMusFeatures.shape)

#%%
# salvar dataframe em .pickle
dfUserAMusFeatures.to_pickle ("./FeatureStore/AudioFeaturesUserA.pickle")

# %%
logging.info('GetUserA_AudioFeatures <<')