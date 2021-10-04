from flask import Flask, jsonify, Response, request
from flask_cors import CORS, cross_origin
import pandas as pd
from ClassifPredicao import Predicao
import os
import pickle
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# iniciando estatísticas
if (os.path.isfile("estatisticas.pickle") == False):
    estatisticas = {}
    estatisticas['MusNaoEncontradaEmAudioFeature'] = 0
    estatisticas["CurteAnaliseConteudoNaobateComAnaliseColab"] = 0
    estatisticas["NaoCurteAnaliseConteudoNaobateComAnaliseColab"] = 0
    estatisticas["NumTotalDePredicoes"] = 0
    estatisticas["PredicoesCorretas"] = 0
    estatisticas["PredicoesFalsoPositivo"] = 0
    estatisticas["PredicoesFalsoNegativo"] = 0
    with open('estatisticas.pickle', 'wb') as arq:
        pickle.dump(estatisticas, arq)
    arq.close()

def updateStats(data):
    estatisticas = pd.read_pickle("estatisticas.pickle")
    estatisticas["NumTotalDePredicoes"] += 1
    if (data['predicao'] == data['gostoReal']):
        estatisticas["PredicoesCorretas"] += 1
    else:
        if (data['predicao'] == 'NaoCurte'):
            estatisticas["PredicoesFalsoNegativo"] = estatisticas.get("PredicoesFalsoNegativo",0) +1
        else:
            estatisticas["PredicoesFalsoPositivo"] = estatisticas.get("PredicoesFalsoPositivo",0)+1
    with open('estatisticas.pickle', 'wb') as arq:
        pickle.dump(estatisticas, arq)

def getStats():
    estatisticas = pd.read_pickle("estatisticas.pickle")
    totalDePredicoes = estatisticas.get("NumTotalDePredicoes");
    predicoesCorretas = estatisticas.get("PredicoesCorretas");
    data = "Total de Predicoes: {}    Predicoes Corretas: {} ({:.0f}%)\nFalsas Predicoes Curto: {}    Falsas Predicoes Nao Curto: {}".format(
        totalDePredicoes, 
        predicoesCorretas,
        (predicoesCorretas*100/totalDePredicoes),
        estatisticas.get("PredicoesFalsoPositivo"), 
        estatisticas.get("PredicoesFalsoNegativo"))
    return data

@app.route('/predicao/', methods=['GET', 'POST'])
@cross_origin()
def rotaPredicao() -> object:
    try:
        if (request.method) == 'GET':
            return jsonify(Predicao())
        if (request.method) == 'POST':
            updateStats(request.json)
            return jsonify(isError=False,
                           message="Success",
                           statusCode=200)
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
