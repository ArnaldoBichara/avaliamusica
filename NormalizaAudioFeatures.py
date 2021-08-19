#############
# Este componente faz normaliza colunas em AudioFeatures
# e em UserAAudioFeatures
#############

#%%
import pandas as pd
import numpy as np
import pickle
import logging
from time import gmtime, strftime

logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('NormalizaAudioFeatures >>')


dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserA.pickle")  


#%% normalizando atributos key, tempo, time_signature e duration_ms entre 0 e 1
def normaliza_minmax(df, valormax):
    return (df - df.min()) / ( valormax - df.min())

dfAudioFeatures[['duration_ms']] = normaliza_minmax(dfAudioFeatures[['duration_ms']], 1000000)
dfAudioFeatures[['key']] = normaliza_minmax(dfAudioFeatures[['key']], 11)
dfAudioFeatures[['tempo']] = normaliza_minmax(dfAudioFeatures[['tempo']],200)
dfAudioFeatures[['time_signature']] = normaliza_minmax(dfAudioFeatures[['time_signature']], 5)

#%% salvando filtrado
dfAudioFeatures.to_pickle('./FeatureStore/AudioFeaturesUserA.pickle')

logging.info('NormalizaAudioFeatures <<')