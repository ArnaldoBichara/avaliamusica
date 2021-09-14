# Vamos analisar a base de vizinhos, a matriz de confusão, se o resultado foi bom.

#%% Importando packages
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import confusion_matrix



CandCurto    =  pd.read_pickle ("./FeatureStore/MusCandidatasCurte.pickle")  
CandNaoCurto =  pd.read_pickle ("./FeatureStore/MusCandidatasNaoCurte.pickle")  
IntersCurto    =pd.read_pickle ("./FeatureStore/MusInterseccaoVizinhoscomA.pickle")  

inter = np.sort(IntersCurto['interpretacao'].values)
#%%
CandCurto.loc[CandCurto['interpretacao'].str.contains("SKRILLEX:>")]
#%%
CandNaoCurto.loc[CandNaoCurto['interpretacao'].str.contains("SKRILLEX:>")]
#%%
print(inter)

# %%
