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
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")  
dfUserAbarraAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")  

# filtrando em AudioFeatures, apenas linhas
# com speechiness < 0,6. Mais que isso certamente
# não são músicas, são fala
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['speechiness'] < 0.6]

# filtrando em AudioFeatures apenas linhas com 
# duration_ms > 60000. Não vamos considerar músicas 
# menores que 60 segundos.
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['duration_ms'] > 60000]

#%% removendo coluna loudness já que
# é muito correlacionada com energy e acousticness
dfAudioFeatures.drop(columns=['loudness'], inplace=True)
dfUserAAudioFeatures.drop(columns=['loudness'], inplace=True)
dfUserAbarraAudioFeatures.drop(columns=['loudness'], inplace=True)

print (dfAudioFeatures.shape)
print (dfAudioFeatures.describe())

#%% salvando filtrado
dfAudioFeatures.to_pickle('./FeatureStore/AudioFeatures.pickle')
dfUserAAudioFeatures.to_pickle('./FeatureStore/AudioFeaturesUserACurte.pickle')
dfUserAbarraAudioFeatures.to_pickle('./FeatureStore/AudioFeaturesUserANaoCurte.pickle')

# %%
