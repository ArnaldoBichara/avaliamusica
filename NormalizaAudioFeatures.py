#############
# Este componente faz filtros em AudioFeatures
# e em UserAAudioFeatures
# a partir da análise feita previamente
#############

#%%
import pandas as pd
import numpy as np
import pickle


dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  


#%% normalizando atributos key, tempo, time_signature e duration_ms entre 0 e 1
def normaliza_minmax(df):
    return (df - df.min()) / ( df.max() - df.min())

dfAudioFeatures[['duration_ms']] = normaliza_minmax(dfAudioFeatures[['duration_ms']])
dfAudioFeatures[['key']] = normaliza_minmax(dfAudioFeatures[['key']])
dfAudioFeatures[['tempo']] = normaliza_minmax(dfAudioFeatures[['tempo']])
dfAudioFeatures[['time_signature']] = normaliza_minmax(dfAudioFeatures[['time_signature']])

#%% salvando filtrado
dfAudioFeatures.to_pickle('./FeatureStore/AudioFeatures.pickle')

# %%
