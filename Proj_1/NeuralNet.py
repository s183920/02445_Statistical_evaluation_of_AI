# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:55:23 2020

@author: peter
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression as logreg

df = pd.read_csv("Data_exp5.csv")
X_clas = df.iloc[:,np.array([4,5,6,7])].values
true_class = df["Person"].values



model = logreg(multi_class = "multinomial", solver = "saga", max_iter = 2500)
model.fit(X_clas, true_class)
predictions = model.predict(X_clas)
print(np.mean(predictions == true_class))