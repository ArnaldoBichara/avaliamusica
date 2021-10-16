###################
# Este componente lê as amostras de música e
# monta os espectrogramas, salvando X e y para uso pelos modelos
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
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
logging.info('>> MontaEspectrogramas')

# Obtendo as audio_samples das listas de playlists
#
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

X_spect = np.empty((0,640,128))
y_arr   = np.empty((0))

def getSpectMusicas(X, y, classe):
    contador=0
    dir = 
    # incluindo músicas da playlist
    playlistItems = sp.user_playlist_tracks (user, playlist_id)
    for i, item in enumerate (playlistItems['items']):
        idMusica = item['track']['id']
        amostraMusica = item['track']['preview_url']
        if amostraMusica is not None:
            espectrograma = montaEspectrograma (idMusica, classe)
            X = np.append(X, [espectrograma], axis=0)
            y = np.append(y, [classe], axis=0)
    return X, y

X_spect, y_arr = getSpectMusicas (X_spect, y_arr, 0)
X_spect, y_arr = getSpectMusicas (X_spect, y_arr, 1)
 
print (X_spect.shape)
print (y_arr.shape)
np.savez_compressed('./FeatureStore/AudioEspectrogramas', X_spect, y_arr)


# %%
logging.info('<< MontaEspectrogramas')