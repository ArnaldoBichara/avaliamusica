###################
# Este componente lê e trata as playlists Curto e Não Curto do Usuário A,
# obtendo diretamente do Spotify,
# e salva as audio samples de cada música em arquivo.
# monta os espectrogramas e salva em arquivo, separado pela classe
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
import pathlib
import librosa
from matplotlib import pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import skimage.io

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
def download_amostra(id, url):
    pathlib.Path(f'amostras').mkdir(parents=True, exist_ok=True)
    nome_arq = f'./amostras/{id}'
    if not os.path.exists(nome_arq):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(nome_arq, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        



# monta e retorna espectrograma como numpy array
def montaEspectrograma (id, classe):
    # converte mp3 para espectrograma
    arq_mp3 = f'./amostras/{id}'
    #y, sr = librosa.load(nome_mp3, mono=True, duration=5)
    y, sr = librosa.load(arq_mp3)
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    #skimage.io.imsave(nome_arq, mels)    # save as PNG
    spect = spect.T
    # Normaliza tamanho
    spect = spect[:640, :]
#    print (spect.shape)
    return spect
 
# obtem amostra das Músicas da lista de items fornecida pelo Spotipy
def downloadAmostrasMontaEspectrogramas (X, y, playlistItems, classe):
    for i, item in enumerate (playlistItems['items']):
        idMusica = item['track']['id']
        amostraMusica = item['track']['preview_url']
        if amostraMusica is not None:
            download_amostra(idMusica, amostraMusica)
            espectrograma = montaEspectrograma (idMusica, classe)
            X = np.append(X, [espectrograma], axis=0)
            y = np.append(y, [classe], axis=0)
    return X, y

def getSpectMusicas(user,X, y, playlist_id, classe):
    contador=0
    dfSpect = pd.DataFrame(columns = ["espectrograma", "classe"])
    # incluindo músicas da playlist
    playlistItems = sp.user_playlist_tracks (user, playlist_id)
    X, y = downloadAmostrasMontaEspectrogramas (X, y, playlistItems, classe)
    print (y.shape)
    while playlistItems['next']:
        playlistItems = sp.next(playlistItems)
        X, y = downloadAmostrasMontaEspectrogramas(X, y, playlistItems, classe)
        print (y.shape)
    return X, y
 
X_spect = np.empty((0,640,128))
y_arr   = np.empty((0))

X_spect, y_arr = getSpectMusicas(userA, X_spect, y_arr, IdPlaylistCurto, 1 )
X_spect, y_arr = getSpectMusicas(userA, X_spect, y_arr, IdPlaylistNaoCurto, 0) 

print (X_spect.shape)
print (y_arr.shape)
np.savez_compressed('./FeatureStore/AudioEspectrogramas', X_spect, y_arr)


# %%
logging.info('<< GetUserA_AudioSamples')