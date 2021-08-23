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

logging.basicConfig(filename='./Analises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> PreProcColaboracao')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
dfMusUserACurte =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
dfMusUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  



#
dfMusUsers.shape


#
listaMus              = dfMusUsers['interpretacao'].drop_duplicates()
logging.info('listaMus original %s',listaMus.shape)

listaMusUserACurte    = dfMusUserACurte['interpretacao'].drop_duplicates()
listaMusUserANaoCurte = dfMusUserANaoCurte['interpretacao'].drop_duplicates()

# concatenando musicas com as do UserA
listaMus = pd.concat([listaMus, listaMusUserACurte, listaMusUserANaoCurte], ignore_index=True, verify_integrity=True)        
logging.info ('listaMus depois de concatenar musicas do User A %s', listaMus.shape)

#  Removendo duplicados
listaMus.drop_duplicates(inplace=True)
logging.info ('listaMus depois de remover duplicados %s', listaMus.shape)
#
listaMus.head()
#
listaMus.shape
# análise
print (listaMus[listaMus.str.contains("The Beatles:>let", na= False, case=False)].to_string(index=False))

# salvando dataset
listaMus.to_pickle ("./FeatureStore/DominioDasMusicas.pickle")

# pergunta: quais músicas do dicionário estão no spotify?

# conectando no spotify
scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

#%%
# rotina para verificar se uma interpretação está no spotify
def verMusSpotify (interpretacao):
    encontrada = False;
    interp_splited = interpretacao.split(':>')
    artista = interp_splited[0]
    musica = interp_splited[1]
    item = sp.search('"'+musica+'"' +' artist:'+'"'+artista+'"', limit=1, market='BR')
    return len(item.get('tracks').get('items'))==1


#    item = sp.search('track:'+'"'+musica+'"' +' artist:'+'"'+artista+'"', limit=1, market='BR',type='track')
    print (len(item.get('tracks').get('items')))
    return encontrada

teste = verMusSpotify("Milton Nascimento:>Nuvem Cigana")
print (teste)

#%%

logging.info('<< PreProcColaboracao')