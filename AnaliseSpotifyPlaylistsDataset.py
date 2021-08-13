#%% Importando packages
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import logging
from time import gmtime, strftime
#%%
# iniciando logging de métricas
logging.basicConfig(filename='./Resultado das Análises/avaliamusica.log', 
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
#%%
logging.info ("teste")                  

#%% lendo dataset
dfMusicasUser = pd.read_csv('./datasets/spotify_playlists_dataset.csv', 
                            sep=',', escapechar='\\',
                            nrows=1.1e9,
                            error_bad_lines=False)

#%%
logging.info ("teste %s", dfMusicasUser.describe())
# %%
print(dfMusicasUser.tail())
print(dfMusicasUser.count())
#%% Limpeza: renomeando algumas colunas e removendo coluna playlistname
dfMusicasUser.columns = ['userid', 'artista', 'musica', 'playlistname']
del dfMusicasUser ['playlistname']
print(dfMusicasUser.head())

# %% Limpeza: removendo musicas em branco
if dfMusicasUser['musica'].isnull().sum():
    dfMusicasUser = dfMusicasUser.dropna(axis = 0, subset = ['musica'])
# %% Limpeza: removendo linhas duplicadas 
dfMusicasUser.drop_duplicates(inplace = True)
dfMusicasUser.reset_index(drop=True)
#%%
# %% salvando dataset neste ponto.
#dfMusicasUser.to_pickle ("./arquivos intermediarios/dfMusicasUser.pickle")




# %% Vamos pegar músicas do Beto Guedes, se houver, para comparar
# com o que temos no spotify600k
print (dfMusicasUser[dfMusicasUser['artista'].str.contains("Milton Nascimento", na= False, case=False)][['artista', 'musica']].to_string(index=False))

#%% procurando uma música determinada: Norwegian Wood para ver os artistas
dfNorwegian = dfMusicasUser[dfMusicasUser['musica'].str.contains("Viola Enluarada", na= False, case=False)]
#%%
print (dfNorwegian[['artista']].sort_values(by=['artista']).to_string(index=False))


# %%
print (dfMusicasUser[dfMusicasUser['artista'].str.contains("Bob Marley & The Wailers", na= False, case=False)][['artista', 'musica']].to_string(index=False))

# %%
# %% Recuperando dfMusicasUser
#
dfMusicasUser = pd.read_pickle("./arquivos intermediarios/Users_MusCurte.pickle")
print (dfMusicasUser.head())

# %%
