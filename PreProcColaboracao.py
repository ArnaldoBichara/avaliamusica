# este componente prepara alguns datasets usados pelo
# algoritmo de Colaboração de Users


#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging

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
#
# rotina para verificar se uma interpretação está no spotify
def verMusSpotify (interpretacao):
    encontrada = False;
    interp_splited = encontrada.split(':>')
    artista = interp_splited[0]
    musica = interp_splited[1]
    return encontrada

logging.info('<< PreProcColaboracao')