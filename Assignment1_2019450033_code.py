# matplotlib practice

import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-10, 10, 30)
y = np.random.uniform(-10, 10, 30)

a = -0.8
b = 0.7
y_linear = a * x + b

x_red = []
y_red = []

x_blue = []
y_blue = []

for i in range(30):
    if y_linear[i] <= y[i]:
        x_red.append(x[i])
        y_red.append(y[i])
    else:
      x_blue.append(x[i])
      y_blue.append(y[i])

plt.scatter(x_blue, y_blue, c='blue', marker='^', label='Below the Line')
plt.scatter(x_red, y_red, c='red', marker='o', label='Above the Line')

plt.plot(x, y_linear, 'g-', label=f'y = {a}x + {b}')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Random Data Points with Given Line')
plt.legend()
plt.show()
