import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups-61ce93eb-d671-42db-a5e1-005e0a176393.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

regressor = LinearRegression()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print(Y_pred)

for i, (pred, act) in enumerate(zip(Y_pred, Y_test)):
    print(f"Predicted: {pred:.2f}, Actual: {act:.2f}")

print(regressor.coef_)
print(regressor.intercept_)