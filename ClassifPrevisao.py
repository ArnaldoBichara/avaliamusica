###################
# Este componente executa a classificação entre Curtir ou Não Curtir uma música
# a partir do modelo contentbased escolhido durante o treinamento
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
import sys
import os


logging.basicConfig(filename='./Analises/Classificacao.log', 
                    level=logging.INFO,
                    format='%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S'
                    )
logging.info('\n>> ClassifRandomForest')

#lendo modelo e datasets
modelo = pd.read_pickle ("./FeatureStore/modeloClassif.pickle")
dominioAudioFeatures = pd.read_pickle ("./FeatureStore/DominioAudioFeatures.pickle")
musCandidatasCurte = pd.read_pickle ("./FeatureStore/MusCandidatasCurte.pickle")
musCandidatasNaoCurte = pd.read_pickle ("./FeatureStore/MusCandidatasNaoCurte.pickle")

# preparando estatísticas interessantes
if (os.path.isfile("./Analises/estatisticas.pickle")):
  estatisticas = pd.read_pickle ("./Analises/estatisticas.pickle")
else:
  estatisticas = {}
  estatisticas['MusNaoEncontradaEmAudioFeature'] = 0
  estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"] = 0
  estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"] = 0

#%%

# encontra música candidata de tipo (Curte ou NaoCurte)
def EncontraCandidata (tipo):
  if (tipo=='Curte'):
    musCandidatas = musCandidatasCurte
    predicao_esperada = 1
    estat = "CurteAnaliseConteudoNaobateComAnaliseColab"
  else:
    musCandidatas = musCandidatasNaoCurte
    predicao_esperada = 0
    estat = "NaoCurteAnaliseConteudoNaobateComAnaliseColab"
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

    dados_predicao = audioFeaturesMusCand.drop(columns=['id_musica']).to_numpy()
    # fazendo classificação por conteúdo
    label_predicao = modelo.predict(dados_predicao)
    if (label_predicao[0]==[predicao_esperada]):
      encontrou = True
    else:
      estatisticas[estat] = estatisticas.get(estat, 0) +1
      logging.info ("%s - avaliacao nao bate para: %s", tipo, musCandidata['interpretacao'])

  return musCandidata

musCandidataCurte = EncontraCandidata ('Curte')
with open('./FeatureStore/musicaCandidataCurte.pickle', 'wb') as arq:
    pickle.dump(musCandidataCurte, arq)
logging.info ("musicaCandidataCurte %s", musCandidataCurte['interpretacao'])
print ('música candidata a Curtir:', musCandidataCurte['interpretacao'])

musCandidataNaoCurte = EncontraCandidata ('NaoCurte')
with open('./FeatureStore/musicaCandidataNaoCurte.pickle', 'wb') as arq:
    pickle.dump(musCandidataNaoCurte, arq)
logging.info ("musicaCandidataNaoCurte %s", musCandidataNaoCurte['interpretacao'])
print ('música candidata a Não Curtir:', musCandidataNaoCurte['interpretacao'])

with open('./Analises/estatisticas.pickle', 'wb') as arq:
    pickle.dump(estatisticas, arq)
logging.info ("musicas nao encontradas %s", estatisticas["MusNaoEncontradaEmAudioFeature"])
logging.info ("Curte: Analise por conteudo nao bate com Colab %s", estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"])
logging.info ("Nao Curte: Analise por conteudo nao bate com Colab %s", estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"])

logging.info('\n<< ClassifRandomForest')
