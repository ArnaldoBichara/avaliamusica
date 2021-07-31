#
# analisa dfMusicasUser, previamente salvo em .picle
#
#%% Importando packages
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle

#%% restaurando dfMusicasUser
dfMusicasUser = pd.read_pickle("dfMusicasUser.pickle")

#####
# Análise de número de vezes que uma interpretação é tocada por user
#####

#%% histograma
ax = dfMusicasUser.plot.hist(by='nVezesTocada', bins=100, alpha=0.5)
ax = dfMusicasUser.plot.hist(by='nVezesTocada', bins=100, alpha=0.5, range=(0,25))

#%% analisando quantos users mantemos se filtrarmos por musicas tocadas >1, >2, >3, 100 vezes
# verificando também quantas músicas sobram por user
#%%
print(dfMusicasUser[dfMusicasUser['nVezesTocada']>=2].groupby('userid').count()['musica'])
#%%
print(dfMusicasUser[dfMusicasUser['nVezesTocada']>=3].groupby('userid').count()['musica'])
#%%
print(dfMusicasUser[dfMusicasUser['nVezesTocada']>=4].groupby('userid').count()['musica'])
#%%
print(dfMusicasUser[dfMusicasUser['nVezesTocada']>=8].groupby('userid').count()['musica'])
#%%
print(dfMusicasUser[dfMusicasUser['nVezesTocada']>=100].groupby('userid').count()['musica'])


#%% conclusão: filtrando em músicas tocadas ao menos 4 vezes. 
# isso ajuda a certificar que o usuário gosta da música e não diminui muito o número de usuários
# nem a lista de músicas por usuário
dfMusicasUser=dfMusicasUser[dfMusicasUser['nVezesTocada']>=4]

#%% salvando nMusicasUser filtrado
dfMusicasUser.to_pickle("dfMusicasUserFiltrado.pickle")
dfMusicasUser

#%% restaurando dfMusicasUser
dfMusicasUser = pd.read_pickle("dfMusicasUserFiltrado.pickle")

#%% filtros!!!


#%% FILTROS A APLICAR
# música com apenas um caracter
# remover Músicas (-)(---)(- - -)
# verificar se ainda há coisas como Ã
## a compatibilizar com spotify .. Talvez:
  # remover "" e - e ! de músicas e de artistas 
  # remover qq coisa que tenha ?? no início
  # remover caracter .
  # remover acentos 




#%% listando todas as interpretacoes e salvando em arquivo
#interpretacoes = dfMusicasUser['artista']+' & '+ dfMusicasUser['musica']
#interpretacoes = interpretacoes.unique()
#interpretacoes.sort()
#dfInterpretacoes =pd.DataFrame (interpretacoes,columns=['interpretacoes'])
#dfInterpretacoes.to_excel(r'./LastFMInterpretracoes.xlsx',index=False) 
#%%
dfMusicasUser[['artista','musica']].copy
#%%
dfMusicas = dfMusicasUser[['artista','musica']].copy()
print(dfMusicas.head())
print (dfMusicas.count())

dfMusicas = dfMusicas.drop_duplicates(subset=['musica'])
print (dfMusicas.count())

# %%
dfMusicas = dfMusicas.applymap(lambda x: x.encode('unicode_escape').
                 decode('utf-8') if isinstance(x, str) else x)
dfMusicas.to_excel(r'./LastFMMusicas.xlsx',index=False) 
# %%
