
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
#%%
dfAudioFeatures.hist(bins=100, figsize=(18,16))
plt.savefig("./Resultado das Análises/Histograma AudioFeatures.pdf")
dfUserAAudioFeatures.hist(bins=100, figsize=(18,16))
plt.savefig("./Resultado das Análises/Histograma AudioFeatures UserA.pdf")
dfUserAbarradoAudioFeatures.hist(bins=100, figsize=(18,16))
plt.savefig("./Resultado das Análises/Histograma AudioFeatures UserA barra.pdf")
# %% Para cada característica musical
AnalisesTxt = open ('./Resultado das Análises/AudioFeatures.txt', 'w')

print ('\nDanceability:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['danceability'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['danceability'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['danceability'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['danceability'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['danceability'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['danceability'].std()), file= AnalisesTxt)

print ('\nEnergy:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['energy'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['energy'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['energy'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['energy'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['energy'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['energy'].std()), file= AnalisesTxt)

print ('\nMode:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['mode'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['mode'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['mode'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['mode'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['mode'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['mode'].std()), file= AnalisesTxt)

print ('\nSpeechiness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['speechiness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['speechiness'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['speechiness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['speechiness'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['speechiness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['speechiness'].std()), file= AnalisesTxt)

print ('\nAcousticness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['acousticness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['acousticness'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['acousticness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['acousticness'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['acousticness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['acousticness'].std()), file= AnalisesTxt )

print ('\nInstrumentalness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['instrumentalness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['instrumentalness'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['instrumentalness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['instrumentalness'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['instrumentalness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['instrumentalness'].std()), file= AnalisesTxt )

print ('\nValence:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['valence'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['valence'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['valence'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['valence'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['valence'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['valence'].std()), file= AnalisesTxt )

print ('\nLiveness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['liveness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['liveness'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['liveness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['liveness'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['liveness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['liveness'].std()), file= AnalisesTxt )

print ('\nkey:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['key'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['key'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['key'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['key'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['key'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['key'].std()), file= AnalisesTxt )

print ('\ntempo:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['tempo'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['tempo'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['tempo'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['tempo'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['tempo'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['tempo'].std()), file= AnalisesTxt )

print ('\ntime_signature:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['time_signature'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['time_signature'].std()), file= AnalisesTxt )
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['time_signature'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['time_signature'].std()), file= AnalisesTxt )
print ("Users           : Media=","{:.3f}".format(dfAudioFeatures['time_signature'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['time_signature'].std()), file= AnalisesTxt )

AnalisesTxt.close()
# %% descobrindo o valence para algumas músicas
#print (dfUserAAudioFeatures.columns)
print (dfUserAAudioFeatures.loc[dfUserAAudioFeatures['musica'].str.contains("Lilia")])
# %%
dfUserAAudioFeatures.loc[dfUserAAudioFeatures['valence']<0.1][['musica','valence']]
