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

logging.info('NormalizaAudioFeatures <<')