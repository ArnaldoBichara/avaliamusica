#%%
import pandas as pd
import numpy as np
import pickle
# 
dfUserAMusFeatures =  pd.read_pickle ("./FeatureStore/UserA_AudioFeatures.pickle")  
df600kMus_Features =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  

#%%
print(len(df600kMus_Features))
# %% Incluindo músicas do user A no dataset de músicas & features
dfAudioFeatures = pd.concat([df600kMus_Features, dfUserAMusFeatures], ignore_index=True, verify_integrity=True)

# %%
print(len(dfAudioFeatures))

dfAudioFeatures.to_pickle('./FeatureStore/AudioFeatures.pickle')

# %%
