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


logging.basicConfig(filename='./Analises/Classificacao.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifRandomForest')

#lendo modelo e datasets
modeloRF = pd.read_pickle ("./FeatureStore/modeloRandomForest.pickle")
dominioAudioFeatures = pd.read_pickle ("./FeatureStore/dominioAudioFeatures.pickle")
musCandidatasCurte = pd.read_pickle ("./FeatureStore/musCandidatasCurte.pickle")
musCandidatasNaoCurte = pd.read_pickle ("./FeatureStore/musCandidatasNaoCurte.pickle")

# preparando estatísticas interessantes
if (os.path.isfile("./Analises/estatisticas.pickle")):
  estatisticas = pd.read_pickle ("./Analises/estatisticas.pickle")
else:
  estatisticas = {}
  estatisticas['MusNaoEncontradaEmAudioFeature'] = 0
  estatisticas["AnaliseConteudoNaobateComAnaliseColab"] = 0

#%%

# encontra música candidata de tipo (Curte ou NaoCurte)
def EncontraCandidata (tipo):
  if (tipo=='Curte'):
    musCandidatas = musCandidatasCurte
    predicao_esperada = 1
  else:
    musCandidatas = musCandidatasNaoCurte
    predicao_esperada = 0
 
  encontrou = False
  while not encontrou:
    # Escolha aleatória de uma música Candidata
    musCandidata = musCandidatas.sample()
    # descobrindo AudioFeatures dessa música
    id_musica = musCandidata['id_musica'].item()
    audioFeaturesMusCand = dominioAudioFeatures.loc[dominioAudioFeatures['id_musica'].str.contains(id_musica)]
    if (audioFeaturesMusCand.empty == True):
      estatisticas["MusNaoEncontradaEmAudioFeature"] = estatisticas.get("MusNaoEncontradaEmAudioFeature", 0) +1
      continue

    dados_predicao = audioFeaturesMusCand.drop(columns=['id_musica'])
    # fazendo classificação por conteúdo
    label_predicao = modeloRF.predict(dados_predicao)
    if (label_predicao[0]==[predicao_esperada]):
      encontrou = True
    else:
      estatisticas["AnaliseConteudoNaobateComAnaliseColab"] = estatisticas.get("AnaliseConteudoNaobateComAnaliseColab", 0) +1

  return musCandidata

musCandidataCurte = EncontraCandidata ('Curte')
with open('./FeatureStore/musicaCandidataCurte.pickle', 'wb') as arq:
    pickle.dump(musCandidataCurte, arq)
logging.info ("musicaCandidataCurte %s", musCandidataCurte['interpretacao'])

musCandidataNaoCurte = EncontraCandidata ('NaoCurte')
with open('./FeatureStore/musicaCandidataNaoCurte.pickle', 'wb') as arq:
    pickle.dump(musCandidataNaoCurte, arq)
logging.info ("musicaCandidataNaoCurte %s", musCandidataNaoCurte['interpretacao'])

with open('./Analises/estatisticas.pickle', 'wb') as arq:
    pickle.dump(estatisticas, arq)
logging.info ("musicas não encontradas %s", estatisticas["MusNaoEncontradaEmAudioFeature"])
logging.info ("Analise por conteudo nao bate com Colab %s", estatisticas["AnaliseConteudoNaobateComAnaliseColab"])

logging.info('\n<< ClassifRandomForest')
