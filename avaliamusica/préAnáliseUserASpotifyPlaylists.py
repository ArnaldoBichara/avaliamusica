
#%% Importando packages
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

#  obtendo id das playlists Curto e Não Curto do user
#%%
playlists = sp.user_playlists('jmwyg3knv7jv8tu84b19jxu3p')
#%%
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
print ("playlistCurto=%s PlaylistNaoCurto=%s" % (IdPlaylistCurto, IdPlaylistNaoCurto))

#%%
# a partir de um result de uma playlist, lista Ids das Musicas
def getPartialMusicas (result):
    listMusicas=[]
    for i, item in enumerate (result['items']):
        arrayMusica = [item['track']['id'],item['track']['artists'][0]['name'],item['track']['name']]
        listMusicas.append(arrayMusica)
    return listMusicas

# obtendo todos os ids das músicas de uma playlist
def getIdsMusicas(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    listMusicas = getPartialMusicas(results)
    while results['next']:
        results = sp.next(results)
        listMusicas.extend(getPartialMusicas(results))
    return listMusicas

# Obtendo lista de ids das playlists Curto e NaoCurto
listMusicasUserACurte = getIdsMusicas('jmwyg3knv7jv8tu84b19jxu3p', IdPlaylistCurto)
listMusicasUserAnaocurte = getIdsMusicas('jmwyg3knv7jv8tu84b19jxu3p', IdPlaylistNaoCurto)

dfMusicasUserACurte = pd.DataFrame(listMusicasUserACurte,
                             columns=['id_musica','artista','musica'])
dfMusicasUserAnaocurte = pd.DataFrame(listMusicasUserAnaocurte,
                            columns=['id_musica','artista','musica'])

#%%
dfMusicasUserACurte.set_index('id_musica',verify_integrity=True, inplace=True);

dfMusicasUserAnaocurte.set_index('id_musica',verify_integrity=True, inplace=True);

# salvando datasets
#dfMusicasUserACurte.to_pickle ("./arquivos intermediarios/UserA_MusCurte.pickle")
#dfMusicasUserAnaocurte.to_pickle ("./arquivos intermediarios/UserA_MusNaoCurte.pickle")                            


#%%

print ("\nUser A curte:")
print (len(dfMusicasUserACurte.index))
print (dfMusicasUserACurte.tail())
print ("\nUser A não curte:")
print (len(dfMusicasUserAnaocurte.index))
print (dfMusicasUserAnaocurte.tail())


# %% obtendo as audio_features das playlists
#
def getIdsMusicas (result):
    listMusicas=[]
    for i, item in enumerate (result['items']):
        idMusica = item['track']['id']
        listMusicas.append(idMusica)
    return listMusicas


def getMusicas(user,playlist_id_curto, playlist_id_NaoCurto):
    # incluindo músicas da playlist curto
    results = sp.user_playlist_tracks(user,playlist_id_curto)
    musicas = sp.audio_features(getIdsMusicas(results))
    while results['next']:
        results = sp.next(results)
        musicas.extend(sp.audio_features(getIdsMusicas(results)))
    # incluindo músicas da playlist não curto
    results = sp.user_playlist_tracks(user,playlist_id_NaoCurto)
    musicas.extend(sp.audio_features(getIdsMusicas(results)))
    while results['next']:
        results = sp.next(results)
        musicas.extend(sp.audio_features(getIdsMusicas(results)))
    return musicas

UserAMusFeatures = getMusicas('jmwyg3knv7jv8tu84b19jxu3p', IdPlaylistCurto, IdPlaylistNaoCurto)


#%% filtrando item vazio 
UserAMusFeatures = [i for i in UserAMusFeatures if i is not None]
#%% Gerando dataframe de Musicas e Features do User A
dfUserAMusFeatures = pd.DataFrame(UserAMusFeatures)

# renomeando id para id_musica
dfUserAMusFeatures.rename(columns = {'id':'id_musica'},inplace=True)

# Passando o ID como sendo o Index!!!
dfUserAMusFeatures.set_index('id_musica',verify_integrity=True, inplace=True);

# removendo algumas colunas que não nos interessam
dfUserAMusFeatures.drop(columns=['type','uri','track_href','analysis_url'],inplace=True)

#%% 
print(dfUserAMusFeatures.head())
print(dfUserAMusFeatures.tail())

# %%criando dfMusicasUserACurteENaoCurte
dfMusicasUserACurteENaoCurte =  dfMusicasUserACurte.append (dfMusicasUserAnaocurte, verify_integrity=True)

# %% incluir artista e musica em dfUserAMusFeatures
dfUserAMusFeatures = dfUserAMusFeatures.join(dfMusicasUserACurteENaoCurte, how='left')

# %% salvar dataframe em .pickle
#dfUserAMusFeatures.to_pickle ("./arquivos intermediarios/UserAMus&Features.pickle")

# %% recuperando datasets
dfUserAMusFeatures = pd.read_pickle("./arquivos intermediarios/UserAMus&Features.pickle")
print (dfUserAMusFeatures.head())
# %%
dfMusicasUserACurte = pd.read_pickle ("./arquivos intermediarios/UserA_MusCurte.pickle")
dfMusicasUserAnaocurte = pd.read_pickle ("./arquivos intermediarios/UserA_MusNaoCurte.pickle") 
print (dfMusicasUserACurte.head())
# %%
