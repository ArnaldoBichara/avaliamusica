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
import logging
from time import gmtime, strftime

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> FiltraAudioFeatures')

dfAudioFeatures      =  pd.read_pickle ("./FeatureStore/DominioAudioFeatures.pickle")  
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")  
dfUserAbarraAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")  

# filtrando em AudioFeatures apenas linhas com 
# duration_ms > 30000. Não vamos considerar músicas 
# menores que 30 segundos.
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['duration_ms'] > 30000]
dfUserAAudioFeatures = dfUserAAudioFeatures[dfUserAAudioFeatures['duration_ms'] > 30000]
dfUserAbarraAudioFeatures = dfUserAbarraAudioFeatures[dfUserAbarraAudioFeatures['duration_ms'] > 30000]
logging.info('AudioFeatures com duration_ms > 30 seg = %s', dfAudioFeatures.shape)
logging.info('UserAAudioFeatures com duration_ms > 30 seg = %s', dfUserAAudioFeatures.shape)
logging.info('UserAbarraAudioFeatures com duration_ms > 30 seg = %s', dfUserAbarraAudioFeatures.shape)

logging.info('AudioFeatures %s', dfAudioFeatures.shape)
logging.info('AudioFeaturesUserA %s', dfUserAAudioFeatures.shape)
logging.info('AudioFeaturesUserAbarra %s', dfUserAbarraAudioFeatures.shape)

# filtrando em AudioFeatures, apenas linhas
# com speechiness < 0,6. Mais que isso certamente
# não são músicas, são fala
logging.info('AudioFeatures antes de filtro = %s', dfAudioFeatures.shape)
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['speechiness'] < 0.6]
logging.info('AudioFeatures com speechiness < 0,6 = %s', dfAudioFeatures.shape)

# removendo coluna loudness já que
# é muito correlacionada com energy e acousticness
dfAudioFeatures.drop(columns=['loudness'], inplace=True, errors='ignore')
dfUserAAudioFeatures.drop(columns=['loudness'], inplace=True, errors='ignore')
dfUserAbarraAudioFeatures.drop(columns=['loudness'], inplace=True, errors='ignore')

logging.info('AudioFeatures sem loudness = %s',      dfAudioFeatures.shape)
logging.info('AudioFeaturesUserA sem loudness = %s', dfUserAAudioFeatures.shape)
logging.info('AudioFeaturesUserAbarra sem loudness = %s', dfUserAbarraAudioFeatures.shape)

# normalizando atributos key, tempo, time_signature e duration_ms entre 0 e 1
def normaliza_minmax(df, valormax):
    return (df - df.min()) / (valormax - df.min())

dfAudioFeatures[['duration_ms']] = normaliza_minmax(dfAudioFeatures[['duration_ms']], 1000000)
dfAudioFeatures[['key']] = normaliza_minmax(dfAudioFeatures[['key']], 11)
dfAudioFeatures[['tempo']] = normaliza_minmax(dfAudioFeatures[['tempo']],200)
dfAudioFeatures[['time_signature']] = normaliza_minmax(dfAudioFeatures[['time_signature']], 5)

dfUserAAudioFeatures[['duration_ms']] = normaliza_minmax(dfUserAAudioFeatures[['duration_ms']], 1000000)
dfUserAAudioFeatures[['key']] = normaliza_minmax(dfUserAAudioFeatures[['key']], 11)
dfUserAAudioFeatures[['tempo']] = normaliza_minmax(dfUserAAudioFeatures[['tempo']],200)
dfUserAAudioFeatures[['time_signature']] = normaliza_minmax(dfUserAAudioFeatures[['time_signature']], 5)

dfUserAbarraAudioFeatures[['duration_ms']] = normaliza_minmax(dfUserAbarraAudioFeatures[['duration_ms']], 1000000)
dfUserAbarraAudioFeatures[['key']] = normaliza_minmax(dfUserAbarraAudioFeatures[['key']], 11)
dfUserAbarraAudioFeatures[['tempo']] = normaliza_minmax(dfUserAbarraAudioFeatures[['tempo']],200)
dfUserAbarraAudioFeatures[['time_signature']] = normaliza_minmax(dfUserAbarraAudioFeatures[['time_signature']], 5)

# Monta arquivo de Features para uso dos algoritmos por conteúdo
dfUserAFeatureSamples = pd.concat([dfUserAAudioFeatures,dfUserAbarraAudioFeatures], ignore_index=True, verify_integrity=True)
dfUserAFeatureSamples = dfUserAFeatureSamples.sample(frac=1)
# removendo linhas que tenham algo com NaN
dfUserAFeatureSamples=dfUserAFeatureSamples.dropna()
logging.info("dfUserAFeatureSamples shape=%s", dfUserAFeatureSamples.shape)
# salvando filtrado
dfAudioFeatures.to_pickle('./FeatureStore/DominioAudioFeatures.pickle')
dfUserAFeatureSamples.to_pickle('./FeatureStore/UserAFeatureSamples.pickle')

logging.info('<< FiltraAudioFeatures')
# %%
