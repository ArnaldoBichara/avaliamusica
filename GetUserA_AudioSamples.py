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
        
# from pydub import AudioSegment
# from scipy.io import wavfile
# from tempfile import mktemp

def montaEspectrograma (id, classe):
    # cria path se não existir
    pathlib.Path(f'./espectrogramas/{classe}').mkdir(parents=True, exist_ok=True)
    nome_arq = f'./espectrogramas/{classe}/{id}.png'
    # converte mp3 para espectograma
    if not os.path.exists(nome_arq):
        nome_mp3 = f'./amostras/{id}'
        #y, sr = librosa.load(nome_mp3, mono=True, duration=5)
        y, sr = librosa.load(nome_mp3)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        mels = librosa.power_to_db(mels, ref=np.max)
        # save as PNG
        skimage.io.imsave(nome_arq, mels)

    # obtem amostra das Músicas da lista de items fornecida pelo Spotipy
def downloadAmostras (playlistItems, classe):
    for i, item in enumerate (playlistItems['items']):
        idMusica = item['track']['id']
        amostraMusica = item['track']['preview_url']
        if amostraMusica is not None:
            download_amostra(idMusica, amostraMusica)
            montaEspectrograma (idMusica, classe)

def getAmostrasMusicas(user,playlist_id, classe):
    # incluindo músicas da playlist
    playlistItems = sp.user_playlist_tracks (user, playlist_id)
    downloadAmostras (playlistItems, classe)
    while playlistItems['next']:
        playlistItems = sp.next(playlistItems)
        downloadAmostras(playlistItems, classe)

getAmostrasMusicas(userA, IdPlaylistCurto, 1)
getAmostrasMusicas(userA, IdPlaylistNaoCurto, 0)

# %%
logging.info('<< GetUserA_AudioSamples')