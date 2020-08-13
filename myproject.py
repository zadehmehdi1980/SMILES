#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Details: https://stackoverflow.com/questions/13081532/return-json-response-from-flask-view

import os
import requests
import boto3
from flask import Flask, request #import main Flask class and request object
from flask import render_template
import json
import sys
import argparse
import cnnPredict
import mlpPredict

homeAdrs =os.getcwd()

import logging
logging.basicConfig(level=logging.INFO)


app = Flask(__name__) #create the Flask app

@app.route('/')
def flask_server_health_check():
    return '''<h1> Health Check, The CNN-model and MLP (4 hidden layers) Models are Ready to Receive Input (SMILES) at Below Link:  </h1>
              <h2> http://localhost:5000/predict?{'text':'CC(=O)OC1=CC=CC=C1C(=O)O'} ---  more details: see request_example.py </h2>

            '''
@app.route('/predict', methods=['GET','POST']) #GET and POST
def predict():

    try:
        req_data = request.get_json()

        text = req_data['text']
	cnnPredict_list = list(cnnPredict.prediction(str(text)))
        mlpPredict_list = list(mlpPredict.prediction(str(text)))
        
        body = {
                  'cnnPredict': cnnPredict_list,
                  'mlpPredict': mlpPredict_list
        
                }
        headers = {
                        'Content-Type': 'text/plain'
                }

        data = {
                  'isBase64Encoded': False,
                  'statusCode': '200',
                  'headers': headers,
                  'body': body
                }

        response = app.response_class(
                                        response=json.dumps(data),
                                        status=200,
                                        mimetype='application/json'
                                    )
        return response
    except Exception as e:
        logging.info("we got the following error as exception/n".format(e.message))

if __name__ == '__main__':
    app.debug = True

# Change Host IP (if needed to valid or network IP) 

    app.run(host='127.0.0.1', debug=True, port='5000')
   
