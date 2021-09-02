#%%
import pandas as pd
import numpy as np
import pickle
# seaborn é um ótimo visualizador sobre o matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image

dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")  
dfUserAbarradoAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")  

# %%
dfAudioFeatures.tail(1)
#%% histograma de duration_ms AudioFeatures
dfAudioFeatures[['duration_ms']].plot.hist(by='duration_ms', 
                               bins=100, alpha=0.5)

#%% correlação entre colunas
columns = dfAudioFeatures.columns
print (columns)
#%% tirando algumas colunas do cálculo de correlação
dfMenosColunas = dfAudioFeatures.iloc[:,3:-1]
matriz_de_correlacao = dfMenosColunas.corr()
sns.heatmap(matriz_de_correlacao)
ax = sns.heatmap(
    matriz_de_correlacao, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.title("Correlação entre parâmetros de AudioFeatures")

plt.show()
#%% correlação de algumas colunas
sns_plot = sns.pairplot(dfAudioFeatures[['loudness','acousticness','energy']].sample(1000),height=2.0)
sns_plot.savefig('pairplot.png')
plt.clf() # limpa pairplot do sns
Image(filename='pairplot.png')
#%%
sns_plot = sns.pairplot(dfAudioFeatures[['danceability','valence']].sample(1000),height=2.0)
sns_plot.savefig('pairplot2.png')
plt.clf() # limpa pairplot do sns
Image(filename='pairplot2.png')


#%% removendo coluna loudness
dfAudioFeatures.drop(columns=['loudness'], inplace=True)
#%%
dfAudioFeatures.describe()



# %%
dfUserAAudioFeatures.describe()

#%%
dfAudioFeatures.describe()

# %%
dfUserAbarradoAudioFeatures.describe

# %%
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['speechiness'] < 0.6]
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['duration_ms'] > 60000]

#%%
def normaliza_minmax(df):
    return (df - df.min()) / ( df.max() - df.min())

print (dfAudioFeatures[['duration_ms']].max())
print (dfAudioFeatures[['duration_ms']].min())
#%%
dfAudioFeatures[['duration_ms']] = normaliza_minmax(dfAudioFeatures[['duration_ms']])
dfAudioFeatures[['key']] = normaliza_minmax(dfAudioFeatures[['key']])
dfAudioFeatures[['tempo']] = normaliza_minmax(dfAudioFeatures[['tempo']])
dfAudioFeatures[['time_signature']] = normaliza_minmax(dfAudioFeatures[['time_signature']])
#%%
print (dfAudioFeatures[dfAudioFeatures['duration_ms']==1])
# %%
# %%
dfAudioFeatures.hist(bins=100, figsize=(18,16))

# %%
