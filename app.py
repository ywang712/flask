# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:45:37 2023

@author: yong
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    num_features = [float(x) for x in request.form.values()]
    final_features = [np.array(num_features)]
    prediction = model.predict(final_features)

    

    return render_template('index.html', prediction_text='Does client do term deposit? {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)