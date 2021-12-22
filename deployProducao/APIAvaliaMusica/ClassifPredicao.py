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

def Predicao( modelo, 
              dominioAudioFeatures, 
              musCandidatasCurte,
              musCandidatasNaoCurte,
              estatCurteAnaliseConteudoNaobateComAnaliseColab,
              estatNaoCurteAnaliseConteudoNaobateComAnaliseColab ) -> dict:


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
  estatisticas = {}
  estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"] = estatCurteAnaliseConteudoNaobateComAnaliseColab
  estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"] = estatNaoCurteAnaliseConteudoNaobateComAnaliseColab

  while not encontrou:
    # Escolha aleatória de uma música Candidata
    musCandidata = musCandidatas.sample()
    # descobrindo AudioFeatures dessa música
    id_musica = musCandidata['id_musica'].item()
    audioFeaturesMusCand = dominioAudioFeatures.loc[dominioAudioFeatures['id_musica'].str.contains(id_musica)]
    if (audioFeaturesMusCand.empty == True):
      # estatisticas["MusNaoEncontradaEmAudioFeature"] = estatisticas.get("MusNaoEncontradaEmAudioFeature", 0) +1
      continue
    dados_predicao = audioFeaturesMusCand.drop(columns=['id_musica']).to_numpy()
    # fazendo classificação por conteúdo
    label_predicao = (modelo.predict(dados_predicao) > 0.5).astype(int)
    if (label_predicao[0]==[predicao_esperada]):
      encontrou = True
    else:
      estatisticas[estat] +=1
      #logging.info ("%s - avaliacao nao bate para: %s", tipo, musCandidata['interpretacao'])

  # salvando estatísticas acumuladas
#  with open('estatisticas.pickle', 'wb') as arq:
#      pickle.dump(estatisticas, arq)

  #logging.info ("musicas nao encontradas {0}".format(estatisticas["MusNaoEncontradaEmAudioFeature"]))
  #logging.info ("Curte: Analise por conteudo nao bate com Colab {0}".format(estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"]))
  #logging.info ("Nao Curte: Analise por conteudo nao bate com Colab {0}".format(estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"]))
  #logging.info ("musicaCandidata %s %s", tipo, musCandidata['interpretacao'].to_string(index=False))

  #logging.info('\n<< ClassifPredicao')

  return {'tipo':tipo, 
          'interpretacao':musCandidata['interpretacao'].to_string(index=False),
          'CurteAnaliseConteudoNaobateComAnaliseColab': estatisticas['CurteAnaliseConteudoNaobateComAnaliseColab'],
          'NaoCurteAnaliseConteudoNaobateComAnaliseColab': estatisticas['NaoCurteAnaliseConteudoNaobateComAnaliseColab']}
  
  