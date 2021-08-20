# vamos analisar o UserA com os Users
# pergunta: as músicas do userA tem alguma correspondência nos Users?

#%% Importando packages
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='./Resultado das Análises/preprocessamento2.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('>> Analisa UserA com Users')

dfMusUsers =  pd.read_pickle ("./FeatureStore/MusUsers.pickle")  
dfMusUserA =  pd.read_pickle ("./FeatureStore/MusUserACurte.pickle")  

#%% vamos achar uma música do UserA
dfMusUserA.tail()

#%% Música 1: Christopher Phillips:>Your Song
# Vamos ver o que temos desse artista em Users
print (dfMusUsers[dfMusUsers['interpretacao'].str.contains("Christopher Phillips", na= False, case=False)].to_string(index=False))
# achamos apenas uma música, e não é a mesma de minha lista.

#%%Vamos ver músicas do Beto Guedes, quem sabe dá mais sorte
print (dfMusUserA[dfMusUserA['interpretacao'].str.contains("Beto Guedes", na= False, case=False)].to_string(index=False))
#%% uma básica: Beto Guedes:>Amor De Índio
print (dfMusUsers[dfMusUsers['interpretacao'].str.contains("Beto Guedes:>Amor De Índio", na= False, case=False)].to_string(index=False))
#%% não tem... vixe. Vamos ver se temos algo do Beto
print (dfMusUsers[dfMusUsers['interpretacao'].str.contains("Beto Guedes", na= False, case=False)]['interpretacao'].to_string(index=False))
#%% bem, achei duas músicas...
#%% vamos ver do Milton, que é mais popular
print (dfMusUserA[dfMusUserA['interpretacao'].str.contains("Milton Nascimento", na= False, case=False)].to_string(index=False))
#%%
print (dfMusUsers[dfMusUsers['interpretacao'].str.contains("Milton Nascimento", na= False, case=False)]['interpretacao'].to_string(index=False))
#%%
logging.info('<< Analisa UserA com Users')

# %%
