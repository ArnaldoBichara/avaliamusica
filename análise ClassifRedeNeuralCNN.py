###################
# Análise de classificação por conteúdo,
# usando Rede Neural Convolucional e
# espectrogramas
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import logging
import librosa
import matplotlib.pyplot as plt
import librosa.display
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

logging.basicConfig(filename='./Analises/processamClassifNeuralNetwork.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
             
audio_data = './amostras/0adjqd1H9YczEJSUPfFvwt'
x , sr = librosa.load(audio_data)

#salvando transformada de fourier 
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
# apresenta espectrograma
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

audio_data = './amostras/4LFH3N76rlcjJyYtkpToZa'
x , sr = librosa.load(audio_data)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

# %%
