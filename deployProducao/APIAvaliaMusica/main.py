from flask import Flask, jsonify, Response
from flask_cors import CORS, cross_origin
import pandas as pd
from ClassifPredicao import Predicao

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predicao', methods=['GET'])
@cross_origin()
def root() -> object:
    try:
        return jsonify(Predicao())
    except:
        return Response("erro", status=404, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
