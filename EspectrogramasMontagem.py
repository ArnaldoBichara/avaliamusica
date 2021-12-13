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
#import skimage.io
from sklearn.model_selection._split import train_test_split
import tensorflow as tf
import random
#%%
logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> MontaEspectrogramas')

# Obtendo as audio_samples das listas de playlists
#
# monta e retorna espectrograma como numpy array
def montaEspectrograma (y, samplingRate, classe):
    # converte mp3 para espectrograma
    spect = librosa.feature.melspectrogram(y=y, sr=samplingRate, n_fft=2048, hop_length=1024)
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

def resampleEAugmentation (arq_mp3):
    # carrega arquivo, convertido para mono, com resampling de 44100hz e duração máxima de 30 seg
    max_mseg = 30000
    samplingRate = 44100
    data = librosa.load (arq_mp3, res_type='kaiser_fast')[0]
    data = librosa.load (arq_mp3, sr=44100, mono=True, duration=max_mseg, res_type='kaiser_fast')[0]

    # fixando o tamanho do espectrograma em tamanho baseado no samplingRate e tempo da amostra
    input_length = samplingRate//1000 * max_mseg
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0,input_length - len(data))), "constant")

    # adicionando ruído
    noise = 0.005 * np.random.randn(len(data))
    data = data + noise
    #%%
    # shift nos dados em até 40%
    # vou retirar isso. Não vejo sentido. Deve estragar o RNN
    #shift_limit_pct = 0.4
    #tam_shift = int(random.random()*0.4*input_length)
    #data = np.roll(data, tam_shift)

    # esticando os dados em 20% no tempo
    rate = 1.2
    data = librosa.effects.time_stretch(data, rate )
    
    return data, samplingRate

def getSpectMusicas(X, y, classe):
    contador=0
    for idMusica in os.listdir (f'./amostrasMusica/{classe}'):
        contador +=1
        arq_mp3 = f'./amostrasMusica/{classe}/{idMusica}'
    #    y, samplingRate = librosa.load(arq_mp3)
        sig, samplingRate = resampleEAugmentation(arq_mp3)
        espectrograma = montaEspectrograma (sig, samplingRate, classe)
        X = np.append(X, [espectrograma], axis=0)
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