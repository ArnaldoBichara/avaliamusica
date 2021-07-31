#%% Importando packages
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
#
# Análise de período de tempo da amostra
#
#%% vamos ver o período de tempo dessa amostra
dftimestamp = pd.read_csv('./datasets/lastfm-1K-users-tracks.tsv',
                          header = None, nrows=2e7, sep='\t',
                          skipinitialspace=True,
                          names = ['userid', 'timestamp', 'musicbrainz-artist-id', 'artista', 'musicbrainz-track-id', 'musica'],
                          usecols = ['timestamp'])
#%%
print(dftimestamp.min())
print(dftimestamp.max())
#%%
dftimestamp['timestamp']= dftimestamp['timestamp'].astype('datetime64')
#%%
dftimestamp.groupby(dftimestamp['timestamp'].dt.year).count().plot(kind="bar")

#
# tratando lasfm-1k e consolidando
#
#%% lendo users dataset através de read_csv
dfusers = pd.read_csv('./datasets/lastfm-1K-users-tracks.tsv',
                          header = None, nrows=2e7, sep='[\t\n]',
                          skipinitialspace=True,
                          names = ['userid', 'timestamp', 'musicbrainz-artist-id', 'artista', 'musicbrainz-track-id', 'musica'],
                          usecols = ['userid', 'artista', 'musica'])

if dfusers['musica'].isnull().sum():
    dfusers = dfusers.dropna(axis = 0, subset = ['musica'])


#%%

dftimestamp.plot.hist()
#%% agregando, por user, artista/musica e número de vezes tocada
dfusers['nVezesTocada']=1
dfMusicasUser = (dfusers.
    groupby (by = ['userid','artista','musica'])
    ['nVezesTocada'].sum().reset_index()
    [['userid','artista','musica','nVezesTocada']]
    )
#%% limpeza: removendo nVezesTocada=1, porque na história de um user, se ouviu apenas uma vez não deve ter gostado
dfMusicasUser = dfMusicasUser[dfMusicasUser['nVezesTocada']!=1]

#%% salvando dfMusicaUser em disco, para recuperar posteriormente
dfMusicasUser.to_pickle("dfMusicasUser.pickle")


# %%
