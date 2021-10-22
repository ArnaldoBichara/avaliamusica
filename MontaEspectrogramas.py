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
from sklearn.model_selection._split import train_test_split

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
    arq_mp3 = f'./amostras/{classe}/{id}'
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

X_spect = np.empty((0,640,128)) # primeiraa dimensão: elementos, seg dimensao: array  de frequências, ter dimensao: array de frames
#y_arr   = np.empty((0,2)) # primeira dimensão: elementos, seg dimensão: array de classes [não curto, curto]
y_arr   = [] # dimensão única, onde classe é curto=1, não curto=0

def getSpectMusicas(X, y, classe):
    contador=0
    for idMusica in os.listdir (f'./amostras/{classe}'):
        contador +=1
        espectrograma = montaEspectrograma (idMusica, classe)
        X = np.append(X, [espectrograma], axis=0)
        """         if (classe==0):
            y = np.append(y, [[1,0]], axis=0) 
        else:
            y = np.append(y, [[0,1]], axis=0) """
            
        y = np.append(y, [classe], axis=0)    
        if contador % 10 == 0:
            print ("processando: ", contador)
    return X, y

X_spect, y_arr = getSpectMusicas (X_spect, y_arr, 0) # músicas Não Curto
X_spect, y_arr = getSpectMusicas (X_spect, y_arr, 1) # músicas Curto

print (X_spect.shape)
print (y_arr.shape)
X_train, X_test, y_train, y_test = train_test_split(X_spect, y_arr, random_state=0, test_size=0.30)

np.savez_compressed("./FeatureStore/AudioEspectrogramasTreinoBin", X_train, y_train)
np.savez_compressed("./FeatureStore/AudioEspectrogramasTesteBin", X_test, y_test)


# %%
logging.info('<< MontaEspectrogramas')