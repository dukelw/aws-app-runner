import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only two features
diabetes_X = diabetes_X[:, [2, 3]]
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
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x1 = diabetes_X_test[:, 0]
x2 = diabetes_X_test[:, 1]
y = diabetes_y_test
y_pred = diabetes_y_pred

ax.scatter(x1, x2, y, color="black", label="Actual Data")

x1_surf, x2_surf = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 100),
    np.linspace(x2.min(), x2.max(), 100),
)
y_surf = regr.intercept_ + regr.coef_[0] * x1_surf + regr.coef_[1] * x2_surf
ax.plot_surface(
    x1_surf, x2_surf, y_surf, color="blue", alpha=0.5, label="Prediction Surface"
)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Target")

plt.legend()
plt.show()
