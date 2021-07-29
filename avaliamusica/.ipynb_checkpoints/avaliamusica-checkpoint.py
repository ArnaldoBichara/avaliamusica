#%% Importando packages
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
# %% lendo users dataset e 
# %% removendo linhas onde não há nome de artista ou nome de músicas
dfUsers = pd.read_table('./datasets/lastfm-1K-users-tracks.tsv',
                          header = None, nrows = 2e7,
                          names = ['userid', 'timestamp', 'musicbrainz-artist-id', 'artista', 'musicbrainz-track-id', 'faixa'],
                          usecols = ['userid', 'artista', 'faixa'])
if dfUsers['artista'].isnull().sum():
    dfUsers = dfUsers.dropna(axis = 0, subset = ['artista'])
if dfUsers['faixa'].isnull().sum():
    dfUsers = dfUsers.dropna(axis = 0, subset = ['faixa'])

# %% olhando o que temos aqui
print (dfUsers.head())
print (dfUsers.describe())
print (dfUsers.columns)



# %% consolidando dados de artista+música por user
dfUsers['nPlays']=1
dfMusPerUser = (dfUsers.groupby (['userid', 'artista','faixa'])
                          .agg({'nPlays':sum}))
# %%
print (dfMusPerUser.head())
#%%
print (dfMusPerUser.describe())
#%%
print (dfMusPerUser.columns) 

#%% sort
dfMusPerUser = (dfMusPerUser.groupby (['userid','artista','faixa','nPlays']))
# %% salvando em arquivo nesse ponto 
dfMusPerUser.to_csv('./datasets/MusPerUser.tsv', sep='\t')      

# %% recuperando dfMusPerUser
dfMusPerUser = pd.read_table('./datasets/MusPerUser.tsv',
                          nrows = 2e6)
# %% 
print (dfMusPerUser.head(20))
# %% teste
g= (dfMusPerUser.groupby('userid'))
print (g.head())
 

# %% estatísticas
g = dfMusPerUser.groupby(['userid','artista','faixa','nPlays'])


# %%

dfMusPerUser.head()
#%%
g.head()

# %%
users_data['plays']=1
mus_por_user = (users_data.groupby (by=['userid', 'artista','faixa'])
                          .agg({'plays':sum}))
                          
                               
        
mus_por_user.head() 
# %%
