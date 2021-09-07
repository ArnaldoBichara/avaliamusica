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

CandCurto    =  pd.read_pickle ("./FeatureStore/MusCandUserACurte.pickle")  
CandNaoCurto =  pd.read_pickle ("./FeatureStore/MusCandUserANaoCurte.pickle")  
IntersCurto    =pd.read_pickle ("./FeatureStore/MusInterseccaoVizinhosComA.pickle")  
IntersNaoCurto =pd.read_pickle ("./FeatureStore/MusInterseccaoVizinhosComAbarra.pickle")
  
dfCandCurto = 
#%%
#logging.info('\n<< ColabAnalisaMusicasCandidatas')

# %%
