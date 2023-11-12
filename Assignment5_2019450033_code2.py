# Stochastic Gradient Method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/content/advertising.csv')
data

df = pd.DataFrame(data)

X = df[['TV', 'Radio', 'Newspaper']]
y = df[['Sales']].values

X_np = df[['TV', 'Radio', 'Newspaper']].values
LenData = len(y)


scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X))

ones_df = pd.DataFrame(np.ones((LenData, 1)), columns=['Bias'])
df = ones_df.merge(X_scaled, left_index=True, right_index=True)
X_np_scaled = df.values


Lr = 0.1
epochs = 1000

w = np.zeros((4, 1))

losses = []

for k in range(epochs):
    y_pred = X_np_scaled @ w
    error = y_pred - y
    loss = 0.5 * np.mean(error**2)
    losses.append(loss)
    gradient = (X_np_scaled.T @ error) / len(y)
    w = w - Lr * gradient

print(w)

y_pred = X_np_scaled @ w

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss(MSE)')
plt.title('Loss Curve')
plt.grid(True)
plt.show()