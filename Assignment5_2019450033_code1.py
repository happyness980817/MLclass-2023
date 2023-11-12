# Batch Gradient Descent

# -*- coding: utf-8 -*-
"""Assignment05_2019450033_code1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LHi114TO3AH-5exwn3oso4W7YGymd-dO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/advertising.csv')
data

df = pd.DataFrame(data)

X = df[['TV', 'Radio', 'Newspaper']]
y = df[['Sales']].values

X_np = df[['TV', 'Radio', 'Newspaper']].values
LenData = len(y)

print(X_np.shape)

print(LenData)



ones_df = pd.DataFrame(np.ones((LenData, 1)), columns=['Bias'])
df = ones_df.merge(X, left_index=True, right_index=True)
X_np = df.values

Lr = 0.00001
epochs = 200

w = np.zeros((4, 1))

losses = []

for k in range(epochs):
    y_pred = X_np @ w
    error = y_pred - y
    loss = 0.5 * np.mean(error**2)
    losses.append(loss)
    gradient = (X_np.T @ error) / LenData
    w = w - Lr * gradient

print(w)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.grid(True)
plt.show()

ones_df = pd.DataFrame(np.ones((LenData, 1)), columns=['Bias'])
df = ones_df.merge(X, left_index=True, right_index=True)
X_np = df.values

Lr = 0.00001
epochs = 400

w = np.zeros((4, 1))

losses = []

for k in range(epochs):
    y_pred = X_np @ w
    error = y_pred - y
    loss = 0.5 * np.mean(error**2)
    losses.append(loss)
    gradient = (X_np.T @ error) / LenData
    w = w - Lr * gradient

print(w)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.grid(True)
plt.show()

Lr = 0.00001
epochs = 1000

w = np.zeros((4, 1))

losses = []

for k in range(epochs):
    y_pred = X_np @ w
    error = y_pred - y
    loss = 0.5 * np.mean(error**2)
    losses.append(loss)
    gradient = (X_np.T @ error) / LenData
    w = w - Lr * gradient


plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss(MSE)')
plt.title('Loss Curve')
plt.grid(True)
plt.show()

Lr = 0.00003
epochs = 1000

w = np.zeros((4, 1))

losses = []

for k in range(epochs):
    y_pred = X_np @ w
    error = y_pred - y
    loss = 0.5 * np.mean(error**2)
    losses.append(loss)
    gradient = (X_np.T @ error) / LenData
    w = w - Lr * gradient

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss(MSE)')
plt.title('Loss Curve')
plt.grid(True)
plt.show()

Lr = 0.0001
epochs = 1000

w = np.zeros((4, 1))

losses = []

for k in range(epochs):
    y_pred = X_np @ w
    error = y_pred - y
    loss = 0.5 * np.mean(error**2)
    losses.append(loss)
    gradient = (X_np.T @ error) / LenData
    w = w - Lr * gradient

print(w)

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss(MSE)')
plt.title('Loss Curve')
plt.grid(True)
plt.show()