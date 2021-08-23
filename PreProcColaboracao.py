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


logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> PreProcColaboracao')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
dfMusUserACurte =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
dfMusUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  

#%%
dfMusUsers.shape

#%%
serDominioMus = dfMusUsers['interpretacao'].drop_duplicates()
logging.info('dfDominioMus original %s',serDominioMus.shape)

serDominioUserACurte    = dfMusUserACurte['interpretacao']
serDominioUserANaoCurte = dfMusUserANaoCurte['interpretacao']

# concatenando musicas com as do UserA
serDominioMus = pd.concat([serDominioMus, serDominioUserACurte, serDominioUserANaoCurte], ignore_index=True, verify_integrity=True)        
serDominioMus.drop_duplicates(inplace=True)
logging.info ('serDominioMus depois de concatenar musicas do User A e remover duplicados %s', serDominioMus.shape)

#%% incluindo coluna de existencia no spotify
dfDominioMus = serDominioMus.to_frame()
dfDominioMus['existeNoSpotify']=False
#%%
dfDominioMus.head
#%%
# rotina para verificar se uma interpretação está no spotify
scope = "user-library-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))


def VerificaSeMusEstáNoSpotify (interpretacao):
    interp_splited = interpretacao.split(':>')
    artista = interp_splited[0]
    musica = interp_splited[1]
    aFazer = True
    while aFazer:
        try:
            item = sp.search('"'+musica+'"' +' artist:'+'"'+artista+'"', limit=1, market='BR')
            encontrou = (len(item.get('tracks').get('items'))==1)
            aFazer = False
            return encontrou;
        except:
            sleep (3) # espera segundos para voltar 
        

i=0
logging.info("início da verificação de quais músicas tem no spotify")
for index, row in dfDominioMus.iterrows():
    res= VerificaSeMusEstáNoSpotify (row['interpretacao'])
    dfDominioMus.loc[index, 'existeNoSpotify'] = res
    i=i+1
#    if ((i % 500)==0):
    logging.info("%s", i)
logging.info("fim da verificação de quais músicas tem no spotify")        


# salvando dataset
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