# este componente prepara alguns datasets usados pelo
# algoritmo de Colaboração de Users


#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from time import sleep
import os



logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> PreProcColaboracao')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
dfMusUserACurte =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
dfMusUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  
if os.path.isfile("./FeatureStore/DominioDasMusicas.pickle"):
  dfDominioMus = pd.read_pickle ("./FeatureStore/DominioDasMusicas.pickle")
else:  
  DominioMus = dfMusUsers['interpretacao'].drop_duplicates()
  DominioUserACurte    = dfMusUserACurte['interpretacao']
  DominioUserANaoCurte = dfMusUserANaoCurte['interpretacao']

  # concatenando musicas com as do UserA
  DominioMus = pd.concat([DominioMus, DominioUserACurte, DominioUserANaoCurte], ignore_index=True, verify_integrity=True)        
  DominioMus.drop_duplicates(inplace=True)
  logging.info ('DominioMus depois de concatenar musicas do User A e remover duplicados %s', DominioMus.shape)
  dfDominioMus = DominioMus.to_frame()
  # incluindo coluna de Id do spotify
  dfDominioMus['idInterpretacao']=''

#%%
print (dfDominioMus.loc[732])
#%%
# rotina para verificar se uma interpretação está no spotify
scope = "user-library-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

def VerificaMusNoSpotify (interpretacao):
    if (isinstance(interpretacao, str) == False):
      return ''
    else:
      interp_splited = interpretacao.split(':>')
      artista = interp_splited[0]
      musica = interp_splited[1]
      aFazer = True
      while aFazer:
          try:
            item = sp.search('"'+musica+'"' +' artist:'+'"'+artista+'"', limit=1, market='BR')
            if (len(item.get('tracks').get('items'))==1):
              aFazer = False
              return item.get('tracks').get('items')[0].get('id')
            else:
              return 'NaoExiste'
          except:
              sleep (3) # espera segundos para voltar 
        

i=0
logging.info("início da verificação de quais músicas tem no spotify")
houveMudanca=False
for index, row in dfDominioMus.iterrows():
    if row['idInterpretacao']=='':  
      resultado= VerificaMusNoSpotify (row['interpretacao'])
      dfDominioMus.loc[[index], 'idInterpretacao'] = resultado
      houveMudanca=True
    i=i+1
    print (index)
    if ((i % 500)==0):
      if houveMudanca:
      # salvando dataset neste ponto
        dfDominioMus.to_pickle ("./FeatureStore/DominioDasMusicas.pickle")
        houveMudanca=False
      #logging.info("%s", i)

logging.info("fim da verificação de quais músicas tem no spotify")        
dfDominioMus.to_pickle ("./FeatureStore/DominioDasMusicas.pickle")



# pergunta: quais músicas do dicionário estão no spotify?

''' 
  const doRequest = async (args, retries) => {
    try {
      const response = await spotifyApi.getMySavedTracks(args);
      return response;
    } catch (e) {
      if (retries > 0) {
        console.error(e);
        await asyncTimeout(
          e.headers['retry-after'] ?
            parseInt(e.headers['retry-after']) * 1000 :
            RETRY_INTERVAL
        );
        return doRequest(args, retries - 1);
      }
      throw e;
    }
  };
# '''
#%%

logging.info('<< PreProcColaboracao')