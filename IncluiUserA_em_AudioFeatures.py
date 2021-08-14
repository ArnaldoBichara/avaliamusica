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
logging.info('IncluUserA_em_AudioFeatures >>')

# 
dfUserAMusFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")  
dfUserAbarraMusFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")  
df600kMus_Features =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  

logging.info ('AudioFeatures antes shape = %s', df600kMus_Features.shape)

# Incluindo músicas do user A no dataset de músicas & features
dfAudioFeatures = pd.concat([df600kMus_Features, dfUserAMusFeatures, dfUserAbarraMusFeatures], ignore_index=True, verify_integrity=True)
dfAudioFeatures.drop_duplicates(inplace=True)

# %%

dfAudioFeatures.to_pickle('./FeatureStore/AudioFeatures.pickle')

# %%
logging.info ('AudioFeatures após incluir userA = %s', dfAudioFeatures.shape)

# %%
logging.info('IncluUserA_em_AudioFeatures <<')
