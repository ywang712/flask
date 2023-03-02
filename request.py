# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:09:09 2023

@author: yong
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

print(r.json())