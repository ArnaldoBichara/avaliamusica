#%%
import pandas as pd
import numpy as np
import pickle
# seaborn é um ótimo visualizador sobre o matplotlib
import seaborn as sea
import matplotlib.pyplot as plt
# %%
dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/UserA_AudioFeatures.pickle")  

# %%
dfAudioFeatures.tail(1)
#%%
#%% histograma de AudioFeatures
dfAudioFeatures[['speechiness']].plot.hist(by='speechiness', 
                               bins=10, alpha=0.5)
#%% histograma de AudioFeatures do UserA
dfUserAAudioFeatures[['speechiness']].plot.hist(by='speechiness', 
                               bins=10, alpha=0.5)
#%%
# filtrando em AudioFeatures, apenas linhas
# com speechiness < 0,6. Mais que isso certamente
# não são músicas, são fala
print (dfAudioFeatures.shape)
dfAudioFeatures = dfAudioFeatures[dfAudioFeatures['speechiness'] < 0.6]
print (dfAudioFeatures.shape)
#%% histograma de duration_ms AudioFeatures
dfAudioFeatures[['duration_ms']].plot.hist(by='duration_ms', 
                               bins=100, alpha=0.5)

#%% normalizando duration_ms entre 0 e 1
def normaliza_minmax(df):
    return (df - df.min()) / ( df.max() - df.min())

dfAudioFeatures[['duration_norm']] = normaliza_minmax(dfAudioFeatures[['duration_ms']])
dfAudioFeatures.drop (columns=['duration_ms'], inplace=True)
dfAudioFeatures[['duration_norm']].plot.hist(by='duration_norm', 
                               bins=100, alpha=0.5)
#%% correlação entre colunas
columns = dfAudioFeatures.columns
#%% tirando algumas colunas do cálculo de correlação
dfMenosColunas = dfAudioFeatures.iloc[:,3:-1]
matriz_de_correlacao = dfMenosColunas.corr()
sea.heatmap(matriz_de_correlacao)
ax = sea.heatmap(
    matriz_de_correlacao, 
    vmin=-1, vmax=1, center=0,
    cmap=sea.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.title("Correlação entre parâmetros de AudioFeatures")

plt.show()
#%% removendo coluna loudness
dfAudioFeatures.drop(columns=['loudness'], inplace=True)
#%%
dfAudioFeatures.describe()