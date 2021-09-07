# Vamos analisar a base de vizinhos, a matriz de confusão, se o resultado foi bom.

#%% Importando packages
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import confusion_matrix

#
logging.basicConfig(filename='./Analises/MusColab.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
#logging.info('\n>> ColabAnalisaMusicasCandidatas')

CandCurto    =  pd.read_pickle ("./FeatureStore/MusCandidatasCurte.pickle")  
CandNaoCurto =  pd.read_pickle ("./FeatureStore/MusCandidatasNaoCurte.pickle")  
IntersCurto    =pd.read_pickle ("./FeatureStore/MusInterseccaoVizinhoscomA.pickle")  

print(IntersCurto['interpretacao'].values)

# %%
