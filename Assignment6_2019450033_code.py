# Hard coding Linear Regression with multiple features
# Weight update by Batch Gradient Descent

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('./iris.csv')
df = df[df['variety'] != 'Setosa']
df['variety'] = df['variety'].replace({'Versicolor': 0, 'Virginica': 1})

X = df[['sepal.width', 'petal.length']].values
Y = df[['variety']].values

LenData = len(Y)
ones = np.ones((LenData, 1))
X = np.concatenate((ones, X), axis=1)

W = np.zeros((3, 1))

Lr = 0.1
epochs = 1000

for k in range(epochs):
    Y_pred = 1 / (1 + np.exp(-X @ W))

    error = Y_pred - Y
    gradient = X.T @ error / LenData

    W = W - Lr * gradient

# Plotting
for i in range(len(Y_pred)):
    color = 'blue' if Y_pred[i] >= 0.5 else 'red'
    plt.scatter(X[i, 1], X[i, 2], c=color)  

# Decision boundary
sepal_width_values = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
petal_length_values = -W[0]/W[2] - (W[1]/W[2]) * sepal_width_values
plt.plot(sepal_width_values, petal_length_values, '-g', label='Decision Boundary')

plt.xlabel('sepal.width')
plt.ylabel('petal.length')
plt.legend()
plt.show()