#############
# Este componente faz filtros em AudioFeatures
# e em UserAAudioFeatures
# a partir da análise feita previamente
#############

#%%
import pandas as pd
import numpy as np
import pickle
# seaborn é um ótimo visualizador sobre o matplotlib
import seaborn as sea
import matplotlib.pyplot as plt

dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/UserA_AudioFeatures.pickle")  

# filtrando em AudioFeatures, apenas linhas
# com speechiness < 0,6. Mais que isso certamente
# não são músicas, são fala
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['speechiness'] < 0.6]

#%% normalizando atributos key, tempo, time_signature e duration_ms entre 0 e 1
def normaliza_minmax(df):
    return (df - df.min()) / ( df.max() - df.min())

dfAudioFeatures[['duration_ms']] = normaliza_minmax(dfAudioFeatures[['duration_ms']])
dfUserAAudioFeatures[['duration_ms']] = normaliza_minmax(dfUserAAudioFeatures[['duration_ms']])

dfAudioFeatures[['key']] = normaliza_minmax(dfAudioFeatures[['key']])
dfUserAAudioFeatures[['key']] = normaliza_minmax(dfUserAAudioFeatures[['key']])

dfAudioFeatures[['tempo']] = normaliza_minmax(dfAudioFeatures[['tempo']])
dfUserAAudioFeatures[['tempo']] = normaliza_minmax(dfUserAAudioFeatures[['tempo']])

dfAudioFeatures[['time_signature']] = normaliza_minmax(dfAudioFeatures[['time_signature']])
dfUserAAudioFeatures[['time_signature']] = normaliza_minmax(dfUserAAudioFeatures[['time_signature']])

#%% removendo coluna loudness já que
# é muito correlacionada com energy e acousticness
dfAudioFeatures.drop(columns=['loudness'], inplace=True)
dfUserAAudioFeatures.drop(columns=['loudness'], inplace=True)

print (dfAudioFeatures.shape)
print (dfAudioFeatures.describe())

#%% salvando filtrado
dfAudioFeatures.to_pickle('./FeatureStore/AudioFeatures.pickle')
dfUserAAudioFeatures.to_pickle('./FeatureStore/UserA_AudioFeatures.pickle')

# %%
