import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# Height (cm)
X = np.array([[149, 147, 155, 158, 163, 165, 168, 170, 171, 175, 178, 180, 183]]).T

# Weight (kg)
y = np.array([[47, 50, 51, 54, 58, 59, 60, 65, 63, 64, 70, 75, 68]]).T

# Visualize data
plt.plot(X, y, "ro")
plt.xlabel("Height (cm)")
plt.xlabel("Weight (kg)")
plt.show()

# Fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Solution found by scikit-learn", regr.intercept_[0], regr.coef_[0][0])

# Preparing the fitting line
b = regr.intercept_[0]
a = regr.coef_[0][0]
x0 = np.linspace(145, 185, 10)
y0 = a * x0 + b

plt.plot(X, y, "ro")
plt.plot(x0.reshape((10, 1)), y0.reshape((10, 1)))
plt.xlabel("Height (cm)")
plt.ylabel("Width (kg)")
plt.show()

y1 = a * 155 + b
y2 = a * 160 + b

print("Predict weight of person with height 155 cm %.2f (kg)" % (y1))
print("Predict weight of person with height 160 cm %.2f (kg)" % (y2))
