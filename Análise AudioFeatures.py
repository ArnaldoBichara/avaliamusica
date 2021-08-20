
#%%
import pandas as pd
import numpy as np
import pickle
# seaborn é um ótimo visualizador sobre o matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
import logging
from time import gmtime, strftime

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>>Análise AudioFeatures')

dfAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeatures.pickle")  
dfUserAAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserACurte.pickle")  
dfUserAbarradoAudioFeatures =  pd.read_pickle ("./FeatureStore/AudioFeaturesUserANaoCurte.pickle")  
#%%
dfAudioFeatures.hist(bins=100, figsize=(18,16))
plt.savefig("./Analises/Histograma AudioFeatures.pdf")
dfUserAAudioFeatures.hist(bins=100, figsize=(18,16))
plt.savefig("./Analises/Histograma AudioFeatures UserA.pdf")
dfUserAbarradoAudioFeatures.hist(bins=100, figsize=(18,16))
plt.savefig("./Analises/Histograma AudioFeatures UserA barra.pdf")
# %% Para cada característica musical
AnalisesTxt = open ('./Analises/AudioFeatures.txt', 'w')

print ('\nDuration_ms', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['duration_ms'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['duration_ms'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['duration_ms'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['duration_ms'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['duration_ms'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['duration_ms'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['duration_ms'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['duration_ms'].std()), file= AnalisesTxt)
print (dfAudioFeatures['duration_ms'].describe(), file=AnalisesTxt)

print ('\nDanceability:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['danceability'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['danceability'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['danceability'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['danceability'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['danceability'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['danceability'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['danceability'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['danceability'].std()), file= AnalisesTxt)
print (dfAudioFeatures['danceability'].describe(), file=AnalisesTxt)

print ('\nEnergy:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['energy'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['energy'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['energy'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['energy'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['energy'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['energy'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['energy'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['energy'].std()), file= AnalisesTxt)
print (dfAudioFeatures['energy'].describe(), file=AnalisesTxt)

print ('\nkey:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['key'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['key'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['key'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['key'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['key'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['key'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['key'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['key'].std()), file= AnalisesTxt )
print (dfAudioFeatures['key'].describe(), file=AnalisesTxt)

print ('\nMode:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['mode'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['mode'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['mode'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['mode'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['mode'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['mode'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['mode'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['mode'].std()), file= AnalisesTxt)
print (dfAudioFeatures['mode'].describe(), file=AnalisesTxt)

print ('\nSpeechiness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['speechiness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['speechiness'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['speechiness'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['speechiness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['speechiness'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['speechiness'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['speechiness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['speechiness'].std()), file= AnalisesTxt)
print (dfAudioFeatures['speechiness'].describe(), file=AnalisesTxt)

print ('\nAcousticness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['acousticness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['acousticness'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['acousticness'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['acousticness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['acousticness'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['acousticness'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['acousticness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['acousticness'].std()), file= AnalisesTxt )
print (dfAudioFeatures['acousticness'].describe(), file=AnalisesTxt)

print ('\nInstrumentalness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['instrumentalness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['instrumentalness'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['instrumentalness'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['instrumentalness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['instrumentalness'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['instrumentalness'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['instrumentalness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['instrumentalness'].std()), file= AnalisesTxt )
print (dfAudioFeatures['instrumentalness'].describe(), file=AnalisesTxt)

print ('\nLiveness:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['liveness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['liveness'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['liveness'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['liveness'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['liveness'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['liveness'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['liveness'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['liveness'].std()), file= AnalisesTxt )
print (dfAudioFeatures['liveness'].describe(), file=AnalisesTxt)

print ('\nValence:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['valence'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['valence'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['valence'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['valence'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['valence'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['valence'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['valence'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['valence'].std()), file= AnalisesTxt )
print (dfAudioFeatures['valence'].describe(), file=AnalisesTxt)

print ('\ntempo:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['tempo'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['tempo'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['tempo'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['tempo'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['tempo'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['tempo'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['tempo'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['tempo'].std()), file= AnalisesTxt )
print (dfAudioFeatures['tempo'].describe(), file=AnalisesTxt)

print ('\ntime_signature:', file=AnalisesTxt)
print ("User A curte    : Media=","{:.3f}".format(dfUserAAudioFeatures['time_signature'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAAudioFeatures['time_signature'].std()), file= AnalisesTxt )
print (dfUserAAudioFeatures['time_signature'].describe(), file=AnalisesTxt)
print ("User A nao curte: Media=","{:.3f}".format(dfUserAbarradoAudioFeatures['time_signature'].mean())," Desvio Padrao=","{:.3f}".format(dfUserAbarradoAudioFeatures['time_signature'].std()), file= AnalisesTxt )
print (dfUserAbarradoAudioFeatures['time_signature'].describe(), file=AnalisesTxt)
print ("600k Mus   : Media=","{:.3f}".format(dfAudioFeatures['time_signature'].mean())," Desvio Padrao=","{:.3f}".format(dfAudioFeatures['time_signature'].std()), file= AnalisesTxt )
print (dfAudioFeatures['time_signature'].describe(), file=AnalisesTxt)

AnalisesTxt.close()
# %% descobrindo o valence para algumas músicas
#print (dfUserAAudioFeatures.columns)
#print (dfUserAAudioFeatures.loc[dfUserAAudioFeatures['musica'].str.contains("Lilia")])
# %%
#dfUserAAudioFeatures.loc[dfUserAAudioFeatures['valence']<0.1][['musica','valence']]

logging.info('<< Análise AudioFeatures')