###################
# Este componente lê e trata as playlists Curto e Não Curto do Usuário A,
# obtendo diretamente do Spotify,
# e salva as audio samples de cada música, juntamente com a classe
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import logging
import requests
import os

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> GetUserA_AudioSamples')

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

# Obtendo as audio_samples das listas de playlists
#
def download_preview(id, url):
    if url is not None:
        nome_arq = './preview/'+id
        if not os.path.exists(nome_arq):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(nome_arq, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

from pydub import AudioSegment
from scipy.io import wavfile
from tempfile import mktemp

def montaEspectrograma (id):
    nome_arq = './preview/'+id    
    # converte mp3 para wav
    mp3_audio = AudioSegment.from_file(nome_arq, format="mp3");
    wname = mktemp('.wav')  # use temporary file    
    mp3_audio.export(wname, format="wav")  # convert to wav
    FS, data = wavfile.read(wname)  # read wav file                        
# obtem preview das Músicas da lista de items fornecida pelo Spotipy
def downloadPreviews (playlistItems):
    for i, item in enumerate (playlistItems['items']):
        idMusica = item['track']['id']
        previewMusica = item['track']['preview_url']
        download_preview(idMusica, previewMusica)
        montaEspectrograma (idMusica)

def getPreviewMusicas(user,playlist_id):
    # incluindo músicas da playlist
    playlistItems = sp.user_playlist_tracks (user, playlist_id)
    downloadPreviews (playlistItems)
    while playlistItems['next']:
        playlistItems = sp.next(playlistItems)
        downloadPreviews(playlistItems)

getPreviewMusicas(userA, IdPlaylistCurto)
getPreviewMusicas(userA, IdPlaylistNaoCurto)

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

# removendo algumas colunas que não nos interessam
dfUserAMusFeatures.drop(columns=['id','type','uri','track_href','analysis_url'],inplace=True)
dfUserAbarraMusFeatures.drop(columns=['id','type','uri','track_href','analysis_url'],inplace=True)

# removendo itens repetidos (diferentes interpretações de um artista\música)
dfUserAMusFeatures.drop_duplicates(inplace=True, ignore_index=True)
dfUserAbarraMusFeatures.drop_duplicates(inplace=True, ignore_index=True)    


#%%
logging.info ("UserAMusFeatures shape %s", dfUserAMusFeatures.shape)
logging.info ("UserAbarraMusFeatures shape %s", dfUserAbarraMusFeatures.shape)

#%%
# salvar dataframe em .pickle
dfUserAMusFeatures.to_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")
dfUserAbarraMusFeatures.to_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")

# %%
logging.info('<< GetUserA_AudioSamples')