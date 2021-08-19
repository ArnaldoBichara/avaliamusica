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


dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserA.pickle")  

logging.info('AudioFeaturesUserA %s', dfUserAAudioFeatures.shape)

#%% removendo coluna loudness já que
# é muito correlacionada com energy e acousticness
dfUserAAudioFeatures.drop(columns=['loudness'], inplace=True)

logging.info('AudioFeaturesUserA sem loudness = %s', dfUserAAudioFeatures.shape)

#%% salvando filtrado
dfUserAAudioFeatures.to_pickle('./FeatureStore/AudioFeaturesUserA.pickle')

logging.info('FiltraAudioFeatures <<')

