# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:55:11 2023

@author: yong
"""

import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('flask.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

num_features = ['age', 'campaign', 'past_days', 'num_of_contacts', 'consumer_price_index', 'consumer_confidence_index']
X = df[num_features]
y = df['y']
model = GradientBoostingClassifier()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))