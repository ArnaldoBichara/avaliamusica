#%%
import pandas as pd
import numpy as np
import pickle
# seaborn é um ótimo visualizador sobre o matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image

#%%
dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")  
dfUserAbarradoAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")  

# %%
dfUserAAudioFeatures.tail(1)
#%% histograma de duration_ms AudioFeatures
dfUserAAudioFeatures[['duration_ms']].plot.hist(by='duration_ms', 
                               bins=100, alpha=0.5)
dfUserAbarradoAudioFeatures[['duration_ms']].plot.hist(by='duration_ms', 
                               bins=100, alpha=0.5)
                               
#%%
print ("User A: duration_ms")
print (dfUserAAudioFeatures['duration_ms'].describe())
print ("\n User A barra: duration_ms")
print (dfUserAbarradoAudioFeatures['duration_ms'].describe())

#%%
dfAudioFeatures.describe()
#%%
dfAudioFeatures.hist(bins=15, alpha=0.5, figsize=(18,16))

#%% histograma do perfil do user A
dfUserAAudioFeatures.hist(bins=15, alpha=0.5, figsize=(18,16))
#%%
dfUserAbarradoAudioFeatures.hist(bins=15, alpha=0.5, figsize=(18,16))


# %%
dfUserAAudioFeatures.describe()

#%%
dfAudioFeatures.describe()

# %%
dfUserAbarradoAudioFeatures.describe

# %%
