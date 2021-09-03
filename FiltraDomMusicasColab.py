# A partir de MusUsers (previamente filtrado) de MusUserA
# vamos montar o domínio das Musicas Colab, ou seja
# para análise colaborativa

#%% Importando packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Analises/preprocessamentoColab.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> FiltraDomMusicasColab')

musUsers        =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
musUserA        =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  
musUserANaoCurte =  pd.read_pickle ("./FeatureStore/MusUserANaoCurte.pickle")  
#

domMusColab = pd.concat([musUserA[['id_musica', 'interpretacao']], musUserANaoCurte[['id_musica', 'interpretacao']],musUsers[['id_musica', 'interpretacao']]], ignore_index=True, verify_integrity=True)        

#
logging.info ("Dominio antes de remover duplicates =%s", domMusColab.shape)
domMusColab.drop_duplicates(subset='interpretacao', inplace=True)
logging.info ("Dominio apos remover duplicates =%s", domMusColab.shape)

# agora que removi as interpretacoes duplicadas, posso remover a coluna interpretacao
domMusColab = domMusColab.drop(columns=['interpretacao'])

# removendo linhas que tenham algo com NaN e removendo o index
domMusColab=domMusColab.dropna()
domMusColab.reset_index(drop=True, inplace=True)

# transformand df em series
domMusColab = domMusColab['id_musica']

# removendo linhas onde id_musica não é uma string válida (por algum motivo isso está acontecendo)
domMusColab= domMusColab[domMusColab.apply(lambda x: isinstance(x,str))]


# ordenando dominioMusicas para tornar acesso mais rápido
domMusColab.sort_values(kind='quicksort', inplace=True)


#
logging.info ("Dominio de musicas para Colab =%s", domMusColab.shape)
# salvando dataset
domMusColab.to_pickle ("./FeatureStore/DominioMusicasColab.pickle")

logging.info('<< FiltraDomMusicasColab')

#

# %%
