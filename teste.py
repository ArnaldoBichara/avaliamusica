###################
# Este componente executa a classificação entre Curtir ou Não Curtir uma música
# a partir do modelo contentbased RandomForest
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
import sys
import os
from sklearn.ensemble import RandomForestClassifier

#lendo modelo e datasets
modeloRF = pd.read_pickle ("./FeatureStore/modeloRandomForest.pickle")
dominioAudioFeatures = pd.read_pickle ("./FeatureStore/DominioAudioFeatures.pickle")
musCandidatasCurte = pd.read_pickle ("./FeatureStore/MusCandidatasCurte.pickle")
musCandidatasNaoCurte = pd.read_pickle ("./FeatureStore/MusCandidatasNaoCurte.pickle")

# preparando estatísticas interessantes
if (os.path.isfile("./Analises/estatisticas.pickle")):
  estatisticas = pd.read_pickle ("./Analises/estatisticas.pickle")
else:
  estatisticas = {}
  estatisticas['MusNaoEncontradaEmAudioFeature'] = 0
  estatisticas["AnaliseConteudoNaobateComAnaliseColab"] = 0


# encontra música candidata de tipo (Curte ou NaoCurte)
musCandidatas = musCandidatasCurte
predicao_esperada = 1
 
# Escolha aleatória de uma música Candidata
musCandidata = musCandidatas.sample()
# descobrindo AudioFeatures dessa música
id_musica = musCandidata['id_musica'].item()
audioFeaturesMusCand = dominioAudioFeatures.loc[dominioAudioFeatures['id_musica'].str.contains(id_musica)]
#    
dados_predicao = audioFeaturesMusCand.drop(columns=['id_musica']).to_numpy()
#
# fazendo classificação por conteúdo
label_predicao = modeloRF.predict(dados_predicao)

print (label_predicao)

# %%
