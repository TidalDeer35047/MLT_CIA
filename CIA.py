import random
import matplotlib.pyplot as plt
import numpy as np


x_points = np.random.rand(1000, 1)
y_points = 8 - 7 * x_points + np.random.randn(1000, 1)
plt.scatter(x_points, y_points, color = "#A020F0")


#SIMPLE LINEAR REGRESSION

'''
x_mean = np.mean(x_points)
y_mean = np.mean(y_points)

n = len(x_points)

numerator = 0
denominator = 0

for i in range(n):
    numerator += (x_points[i] - x_mean) * (y_points[i] - y_mean)
    denominator += (x_points[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)
print(b1, b0)
x_max = np.max(x_points) + 100
x_min = np.min(x_points) - 100

x = np.linspace(x_min, x_max, 1000)

y = b0 + b1 * x


plt.plot(x,y, color = "#30fff0", label = "Simple Linear Regression")
plt.xlim([-5, 15])
plt.ylim([-5, 15])
plt.show()
'''

#GRADIENT DESCENT


'''

w = np.random.randn(1,1)
b = np.random.randn(1,1)

lr = 0.01

for i in range(1000):
    y_pred = x_points @ w + b
    dw = (2/1000)*x_points.T @ (y_pred - y_points)
    db = (2/1000)*np.sum(y_pred - y_points)
    w = w - lr*dw
    b = b - lr*db

# Plot the line  best fit
plt.plot(x_points, x_points*w+b, 'r')
plt.show()

'''

#USING SKLEARN

'''
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_points, y_points)

# print the best fit slope and intercept
print("Best fit slope:", model.coef_)
print("Best fit intercept:", model.intercept_)

plt.plot(x_points, x_points*(model.coef_) + model.intercept_, color = "#30fff0", label = "SKLEARN BEST FIT LINE")
plt.show()
'''