from flask import Flask, jsonify, Response, request
from flask_cors import CORS, cross_origin
import pandas as pd
from ClassifPredicao import Predicao
import os
import pickle
import json
import joblib
from keras.models import load_model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class create_model_MLP():
    pass
if os.path.exists('modeloClassif.h5'):
    #pipeline mlp
    modelo = joblib.load('modeloClassif.pickle')
    modelo.named_steps['mlp'].model = load_model('modeloClassif.h5')
else:
    modelo              = pd.read_pickle ("modeloClassif.pickle")
    
dominioAudioFeatures  = pd.read_pickle ("DominioAudioFeatures.pickle")
musCandidatasCurte    = pd.read_pickle ("MusCandidatasCurte.pickle")
musCandidatasNaoCurte = pd.read_pickle ("MusCandidatasNaoCurte.pickle")

# iniciando estatísticas
if (os.path.isfile("estatisticas.pickle") == False):
    estatisticas = {}
    estatisticas["MusNaoEncontradaEmAudioFeature"] = 0
    estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"] = 0
    estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"] = 0
    estatisticas["predicoes"] = 0
    estatisticas["predicoescorretas"] = 0
    estatisticas["falsospositivos"] = 0
    estatisticas["falsosnegativos"] = 0
    with open('estatisticas.pickle', 'wb') as arq:
        pickle.dump(estatisticas, arq)
    
def updateStats(data):
    estats = pd.read_pickle("estatisticas.pickle")
    estats["predicoes"] += 1
    if (data['predicao'] == data['gostoReal']):
        estats["predicoescorretas"] += 1
    else:
        if (data['predicao'] == 'NaoCurte'):
            estats["falsosnegativos"] = estats.get("falsosnegativos",0) +1
        else:
            estats["falsospositivos"] = estats.get("falsospositivos",0)+1
    with open('estatisticas.pickle', 'wb') as arq:
        pickle.dump(estats, arq)

def getStats():
    estats = pd.read_pickle("estatisticas.pickle")
    totalDePredicoes = estats.get("predicoes");
    predicoesCorretas = estats.get("predicoescorretas");
    data = "Total de Predicoes: {}    Predicoes Corretas: {} ({:.0f}%)\nFalsas Predicoes Curto: {}    Falsas Predicoes Nao Curto: {}".format(
        totalDePredicoes, 
        predicoesCorretas,
        (predicoesCorretas*100/totalDePredicoes),
        estats.get("falsospositivos"), 
        estats.get("falsosnegativos"))
    return data

@app.route('/predicao/', methods=['GET', 'POST'])
@cross_origin()
def rotaPredicao() -> object:
    try:
        if (request.method) == 'GET':
            return jsonify(Predicao( modelo, 
                                     dominioAudioFeatures, 
                                     musCandidatasCurte,
                                     musCandidatasNaoCurte))
        if (request.method) == 'POST':
            updateStats(request.json)
            data = getStats()
            return jsonify(texto=data)
    except:
        return Response("erro", status=404, mimetype='application/json')

@app.route('/stats/', methods=['GET'])
@cross_origin()
def rotaStats() -> object:
    try:
        data = getStats()
        return jsonify(texto=data)
    except:
        return Response("erro", status=404, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
