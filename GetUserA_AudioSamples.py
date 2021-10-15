###################
# Este componente lê e trata as playlists Curto e Não Curto do Usuário A,
# obtendo diretamente do Spotify,
# e salva as audio samples de cada música em arquivo.
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
def download_amostra(id, url):
    if url is not None:
        nome_arq = './amostras/'+id
        if not os.path.exists(nome_arq):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(nome_arq, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

# from pydub import AudioSegment
# from scipy.io import wavfile
# from tempfile import mktemp

''' def montaEspectrograma (id):
    nome_arq = './amostras/'+id    
    # converte mp3 para wav
    mp3_audio = AudioSegment.from_file(nome_arq, format="mp3");
    wname = mktemp('.wav')  # use temporary file    
    mp3_audio.export(wname, format="wav")  # convert to wav
    FS, data = wavfile.read(wname)  # read wav file   '''                      
# obtem amostra das Músicas da lista de items fornecida pelo Spotipy
def downloadAmostras (playlistItems):
    for i, item in enumerate (playlistItems['items']):
        idMusica = item['track']['id']
        amostraMusica = item['track']['preview_url']
        download_amostra(idMusica, amostraMusica)
        #montaEspectrograma (idMusica)

def getAmostrasMusicas(user,playlist_id):
    # incluindo músicas da playlist
    playlistItems = sp.user_playlist_tracks (user, playlist_id)
    downloadAmostras (playlistItems)
    while playlistItems['next']:
        playlistItems = sp.next(playlistItems)
        downloadAmostras(playlistItems)

getAmostrasMusicas(userA, IdPlaylistCurto)
getAmostrasMusicas(userA, IdPlaylistNaoCurto)

# %%
logging.info('<< GetUserA_AudioSamples')