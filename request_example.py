#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Mehdi (Nimahm@gmail.com)
import json
import requests
import os
import time

url = 'http://localhost:5000/predict'


data = {"text": " Cc1ccc(/C=C2\C(=O)NC(=O)N(Cc3ccccc3Cl)C2=O)o1"}


start_time = time.time()
print("inputdata",data)

response = requests.post(url, json=data)
print(response)
print(type(response.content))
print(response.content)
# print(json.loads(response.content)['body'])
print("--- %s seconds ---" % (time.time() - start_time))

