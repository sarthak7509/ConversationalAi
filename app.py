import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import werkzeug
import pickle
import json
import numpy as np
from flask import request
from flask import jsonify
from flask_cors import CORS
from flask import Flask
from coversationalAi import conversationalApi
from intentbot import intentBot

model = None
app = Flask(__name__)

CORS(app)

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

model_path = 'model_weight/model3'

def getmodel(model_path, tokenizer):
    global model
    model = conversationalApi(tokenizer, model_path)

getmodel(model_path, tokenizer)
#this code initialize the model to start predicting
model.predict("hello")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    data = str(message["question"])
    print(data)
    reply = model.predict(data)
    print(reply)
    response = {
        'reply':reply
    }
    return jsonify(response)


@app.route("/")
def index():
    return "<h1>Hello</h1>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
