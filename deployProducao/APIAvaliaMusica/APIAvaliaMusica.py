from flask import Flask, jsonify, request, Response
from security import oauth, token_required
import pickle
import pandas as pd
from ClassifPredicao import Predicao

app = Flask(__name__)

@app.route('/predicao', methods=['POST'])
#@token_required
def root(user:str) -> object:
    try:
        return jsonify(Predicao())
    except:
        return Response("erro", status=404, mimetype='application/json')

@app.route('/security', methods=['POST'])
def security() -> object:
    data = request.authorization
    if not data or 'username' not in data or 'password' not in data:
        return Response("erro", status=404, mimetype='application/json')
    else:
        return oauth(data['username'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
