from flask import Flask, jsonify, Response
import pandas as pd
from ClassifPredicao import Predicao

app = Flask(__name__)

@app.route('/predicao', methods=['POST'])
def root() -> object:
    try:
        return jsonify(Predicao())
    except:
        return Response("erro", status=404, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
