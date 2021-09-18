###################
# Este componente executa a classificação entre Curtir ou Não Curtir uma música
# a partir do modelo contentbased escolhido durante o treinamento
# retorna a música escolhida e o tipo 'Curte', 'NaoCurte'
###################
#%%
# Importando packages
import pandas as pd
import numpy as np
import pickle
import logging
import sys
import os
import random

def Predicao() -> dict:
  logging.info('\n>> ClassifPredicao') 
  logging.basicConfig(filename='./Previsao.log', 
                      level=logging.INFO,
                      format='%(message)s',
                      datefmt='%d/%m/%Y %H:%M:%S'
                      )
  #lendo modelo e datasets
  modelo                = pd.read_pickle ("modeloClassif.pickle")
  dominioAudioFeatures  = pd.read_pickle ("DominioAudioFeatures.pickle")
  musCandidatasCurte    = pd.read_pickle ("MusCandidatasCurte.pickle")
  musCandidatasNaoCurte = pd.read_pickle ("MusCandidatasNaoCurte.pickle")

  # obtendo estatísticas já acumuladas
  if (os.path.isfile("estatisticas.pickle")):
    estatisticas = pd.read_pickle ("estatisticas.pickle")
  else:
    estatisticas = {}
    estatisticas['MusNaoEncontradaEmAudioFeature'] = 0
    estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"] = 0
    estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"] = 0

  # escolhendo aleatoriamente entre tipo 'Curte' ou 'NaoCurte'
  escolhas = ['Curte', 'NaoCurte']
  tipo = random.choice(escolhas)

  # encontra música candidata de tipo (Curte ou NaoCurte)
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

  # salvando estatísticas acumuladas
  with open('estatisticas.pickle', 'wb') as arq:
      pickle.dump(estatisticas, arq)

  logging.info ("musicas nao encontradas {0}".format(estatisticas["MusNaoEncontradaEmAudioFeature"]))
  logging.info ("Curte: Analise por conteudo nao bate com Colab {0}".format(estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"]))
  logging.info ("Nao Curte: Analise por conteudo nao bate com Colab {0}".format(estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"]))
  logging.info ("musicaCandidata %s %s", tipo, musCandidata['interpretacao'].to_string(index=False))

  logging.info('\n<< ClassifPredicao')

  return {'tipo':tipo, 'interpretacao':musCandidata['interpretacao'].to_string(index=False)}
  
  