# Hard coding Multi-Layer Perceptron 
# Weight update by BGD

import numpy as np
from matplotlib import pyplot as plt
import csv

def load_data(filepath):
    with open(filepath, 'r') as file:
        rdr = csv.reader(file)
        next(rdr)  # Skip the header
        # Exclude the 'Setosa' class and load the data
        data = np.array([row for row in rdr if row[4] != 'setosa'])
    return data

def preprocess_data(data):
    X_raw = data[:, :4].astype(float)
    # Label 'Versicolor' as 0 and 'Virginica' as 1
    y_raw = (data[:, 4] == 'Virginica').astype(int)
    return X_raw, y_raw

# log loss function
def log_loss(Y, predictions):
    # 분모 0 으로 나뉘는 것 방지
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNet:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.W1 = np.random.randn(input_size, hidden1_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size)
        self.W3 = np.random.randn(hidden2_size, output_size)
        self.b1 = np.zeros(hidden1_size)
        self.b2 = np.zeros(hidden2_size)
        self.b3 = np.zeros(output_size)

    def forward_propagation(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.output = sigmoid(self.Z3)
        return self.output # predicted y (N * 1 size vector)

    def back_propagation(self, X, Y):
        m = Y.shape[0]
        dZ3 = self.output - Y
        self.delta_W3 = self.A2.T @ dZ3 / m
        self.delta_b3 = np.sum(dZ3, axis=0) / m
        # print(dZ3.shape)
        # print(self.delta_b3.shape)

        dZ2 = (dZ3 @ self.W3.T) * sigmoid_derivative(self.Z2)
        self.delta_W2 = self.A1.T @ dZ2 / m
        self.delta_b2 = np.sum(dZ2, axis=0) / m

        dZ1 = (dZ2 @ self.W2.T) * sigmoid_derivative(self.Z1)
        self.delta_W1 = X.T @ dZ1 / m
        self.delta_b1 = np.sum(dZ1, axis=0) / m

    def update_weights(self, learning_rate):
        self.W3 -= learning_rate * self.delta_W3
        self.b3 -= learning_rate * self.delta_b3
        self.W2 -= learning_rate * self.delta_W2
        self.b2 -= learning_rate * self.delta_b2
        self.W1 -= learning_rate * self.delta_W1
        self.b1 -= learning_rate * self.delta_b1

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward_propagation(X)
            self.back_propagation(X, Y)
            self.update_weights(learning_rate)
            if epoch % 10000 == 0:
                loss = log_loss(Y, predictions)
                print(f"Epoch {epoch}, Loss: {loss}")
                self.plot_activations_with_decision_boundary(X)

    def predict(self, X):
        return self.forward_propagation(X)

    def evaluate_accuracy(self, X, Y):
        predictions = self.predict(X) >= 0.5
        return np.mean(predictions == Y)

    def plot_activations_with_decision_boundary(self, X):
      a1, a2 = self.A2[:, 0], self.A2[:, 1]
      predictions = self.predict(X)
      
      for i in range(len(predictions)):
          color = 'red' if predictions[i] >= 0.5 else 'blue'
          plt.scatter(a1[i], a2[i], color=color)

      # Extracting weights and bias for decision boundary
      w1, w2 = self.W3[0,0], self.W3[1,0]  
      b = self.b3[0]

      # Generate a range of values for a1
      a1_range = np.linspace(a1.min(), a1.max(), 100)

      # Calculate corresponding a2 values for the decision boundary
      a2_boundary = -(w1 / w2) * a1_range - (b / w2)

      plt.plot(a1_range, a2_boundary, color='green')
      plt.title("Activations at Second Hidden Layer with Decision Boundary")
      plt.xlabel("Activation 1 at Second Hidden Layer")
      plt.ylabel("Activation 2 at Second Hidden Layer")
      plt.show()

# Load and preprocess the data
data = load_data('./iris.csv')
X_raw, y_raw = preprocess_data(data)

# Initialize and train the model
model = NeuralNet(4, 3, 2, 1)
model.train(X_raw, y_raw.reshape(-1, 1), epochs=100000, learning_rate=0.01)


# Predict and evaluate accuracy
predicted_y = model.predict(X_raw) >= 0.5
accuracy = np.mean(predicted_y.flatten() == y_raw)
print(f"Accuracy: {accuracy * 100:.2f}%")


