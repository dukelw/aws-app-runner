import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]
print("diabetes X", diabetes_X)

# Number of instances
n = diabetes_X.shape[0]

# Split the data into training and testing sets
diabetes_X_train = diabetes_X[: int(n * 0.8)]
diabetes_X_test = diabetes_X[int(n * 0.8) :]

# Split the target into training and testing sets
diabetes_y_train = diabetes_y[: int(n * 0.8)]
diabetes_y_test = diabetes_y[int(n * 0.8) :]

# Fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficient: \n", regr.coef_)

# The mean squared error
print(
    "Mean squared error: %.2f\n" % mean_squared_error(diabetes_y_test, diabetes_y_pred)
)

# Plot output
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
