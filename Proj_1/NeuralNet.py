# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:55:23 2020

@author: peter
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math

df = pd.read_csv("Data_exp5.csv")
X_clas = df.iloc[:,np.array([4,5,6,7])].values
true_class = df["Person"].values

# print(df["Experiment" == 5'
class ANN:
  def __init__(
    self,
    hidden_units, 
    criterion = nn.CrossEntropyLoss(),
    tolerance = 1e-6,
    optimizer = lambda params: torch.optim.SGD(params, lr = 1e-2),
    
  ):
    self.criterion = criterion
    self.optimizer = optimizer
    self.max_iter = 1000
    self.tolerance = tolerance
    self.hidden_units = hidden_units

  def fit(self, X, y):
    
    X = torch.Tensor(X)
    y = torch.Tensor(y).reshape((-1, 1))
    print(X.shape[1])
    self.model = nn.Sequential(
      #nn.Linear(X.shape[1], y.shape[1])
      nn.Linear(X.shape[1], self.hidden_units),
      nn.Tanh(),
      nn.Linear(self.hidden_units, 10)
    )

    print("Starting training.")
    optimizer = self.optimizer(self.model.parameters())
    old_loss = math.inf
    loss_history = []
    for i in range(self.max_iter):
      optimizer.zero_grad()

      y_hat = self.model(X)
      loss = self.criterion(y_hat, y)
      loss.backward()
      loss_value = loss.item()
      loss_history.append(loss_value)

      p_delta_loss = np.abs(loss_value - old_loss) / old_loss
      if p_delta_loss < self.tolerance: break
      old_loss = loss_value
      
      optimizer.step()
    print("Training done.")
    plt.plot(loss_history)
    plt.show()
    
  def predict(self, X):
    X = torch.Tensor(X)
    y = self.model.forward(X)
    return y.detach().numpy()

model = ANN(5)
model.fit(X = X_clas, y = true_class)
