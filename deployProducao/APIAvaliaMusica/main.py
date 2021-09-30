from flask import Flask, jsonify, Response, request
from flask_cors import CORS, cross_origin
import pandas as pd
from ClassifPredicao import Predicao

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predicao/', methods=['GET', 'POST'])
@cross_origin()
def root() -> object:
    try:
         if (request.method) == 'GET':
            return jsonify(Predicao())
         if (request.method) == 'POST':
            data = request.form #dicionário contendo dados do post
            return jsonify(isError= False,
                    message= "Success",
                    statusCode= 200,
                    data= data)
    except:
        return Response("erro", status=404, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
