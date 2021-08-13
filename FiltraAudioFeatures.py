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

logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('FiltraAudioFeatures >>')


dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")  
dfUserAbarraAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")  

# filtrando em AudioFeatures, apenas linhas
# com speechiness < 0,6. Mais que isso certamente
# não são músicas, são fala
logging.info('AudioFeatures antes de filtro = %s', dfAudioFeatures.shape)
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['speechiness'] < 0.6]
logging.info('AudioFeatures com speechiness < 0,6 = %s', dfAudioFeatures.shape)

# filtrando em AudioFeatures apenas linhas com 
# duration_ms > 60000. Não vamos considerar músicas 
# menores que 60 segundos.
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['duration_ms'] > 40000]
dfUserAAudioFeatures = dfUserAAudioFeatures[dfUserAAudioFeatures['duration_ms'] > 40000]
dfUserAbarraAudioFeatures = dfUserAbarraAudioFeatures[dfUserAbarraAudioFeatures['duration_ms'] > 40000]
logging.info('AudioFeatures com duration_ms > 40 seg = %s', dfAudioFeatures.shape)
logging.info('UserAAudioFeatures com duration_ms > 40 seg = %s', dfUserAAudioFeatures.shape)
logging.info('UserAbarraAudioFeatures com duration_ms > 40 seg = %s', dfUserAbarraAudioFeatures.shape)

#%% removendo coluna loudness já que
# é muito correlacionada com energy e acousticness
dfAudioFeatures.drop(columns=['loudness'], inplace=True)
dfUserAAudioFeatures.drop(columns=['loudness'], inplace=True)
dfUserAbarraAudioFeatures.drop(columns=['loudness'], inplace=True)

logging.info('AudioFeatures sem loudness = %s', dfAudioFeatures.shape)
logging.info('AudioFeaturesUserACurte sem loudness = %s', dfUserAAudioFeatures.shape)
logging.info('AudioFeaturesUserANaoCurte sem loudness = %s', dfUserAbarraAudioFeatures.shape)

#%% salvando filtrado
dfAudioFeatures.to_pickle('./FeatureStore/AudioFeatures.pickle')
dfUserAAudioFeatures.to_pickle('./FeatureStore/AudioFeaturesUserACurte.pickle')
dfUserAbarraAudioFeatures.to_pickle('./FeatureStore/AudioFeaturesUserANaoCurte.pickle')

logging.info('FiltraAudioFeatures <<')

