#%% Importando packages
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import csv
#%% lendo users dataset através de read_csv
dfSpotMusicas = pd.read_csv('./datasets/spotify600k_tracks.csv', 
                            nrows=108e6,
                            usecols = ['id','artists','name','duration_ms','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo', 'time_signature'],
                            #quoting = csv.QUOTE_NONE
                            )
dfSpotMusicas.rename(columns = ({'name':'musica'}), inplace=True)                            
# %%
print(dfSpotMusicas.shape)
print(dfSpotMusicas.head())

# %% df info
dfSpotMusicas.info()
#%%
type(dfSpotMusicas['artists'])
# %% salvando lista de músicas em excel
df = dfSpotMusicas[['artists','musica']].copy()
print (df.count())

df = df.drop_duplicates(subset=['musica'], inplace=True, ignore_index=True)
print (df.count())
df = df.sort_values(by=['musica'])

df = df.applymap(lambda x: x.encode('unicode_escape').
                 decode('utf-8') if isinstance(x, str) else x)
df.to_excel(r'./SpotifyMusicas.xlsx',index=False) 


# %% salvando dfSpotMusicas
dfSpotMusicas.to_pickle('./SpotifyMusicas.pickle')

# %% Recuperando dataframe
#
dfSpotMusicas= pd.read_pickle("./SpotifyMusicas.pickle")

# 
# Filtro: transformando lista de artists em único (primeiro) artista
#%% 
def obtem_primeiro_artista (data): # Clean the data because it are messy strings:
    new_data = []
    # para cada artists do df
    for i in data.values:
        # obtém primeiro artista da lista
        j = i[0].split(',')[0]
        # limpa string e inclui na lista
        new_data.append(j.replace('[', '').replace("'", '').replace(']', '').strip())
        # adiciona à lista
    return new_data

dfSpotMusicas['artista'] = obtem_primeiro_artista(dfSpotMusicas[['artists']])
dfSpotMusicas = dfSpotMusicas.drop(['artists'], axis=1)

# %%
dfSpotMusicas.set_index('id',verify_integrity=True, inplace=True);
dfSpotMusicas.head()

# %% salvando filtrado

# %% salvando dfSpotMusicas
dfSpotMusicas.to_pickle('./SpotifyMusicasFiltrado.pickle')

# %% recuperando musicas filtradas
#
dfSpotMusicas= pd.read_pickle("./SpotifyMusicasFiltrado.pickle")
# %% Verificando uma das músicas curtidas de A: 7JSR685hpFDqvTSNHeSKjL
print(dfSpotMusicas[dfSpotMusicas['musica'].str.contains("60 &", na= False, case=False)][['artista', 'musica']].to_string(index=False))

# %% obtendo músicas do Beto Guedes, por exemplo
#
print(dfSpotMusicas[dfSpotMusicas['artista'].str.contains("Beto Guedes")][['artista', 'musica']].to_string(index=False))
# %% obtendo música Viola Enluarada
#
print(dfSpotMusicas[dfSpotMusicas['musica'].str.contains("Viola Enluarada", na= False, case=False)][['artista', 'musica']].to_string(index=False))

# %%
