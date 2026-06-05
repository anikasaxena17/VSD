import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('studentscores-1fef94ba-27e1-4fab-a7ad-56867b8fb5a1.csv')
plt.scatter(dataset['Hours'], dataset['Scores'])
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
class Model():
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def predict(self, X):
        return X.dot(self.slope) + self.const
    
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.slope = np.zeros(self.n)
        self.const = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self
    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = (-2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = (-2 * np.sum(self.Y - Y_pred)) / self.m
        self.slope -= self.learning_rate * dW
        self.const -= self.learning_rate * db
        return self
    

model = Model()
model.fit(X, Y)

print(model.slope, model.const)
Y_pred = model.predict(X)
print(Y_pred)
plt.scatter(dataset['Hours'], dataset['Scores'])
plt.scatter(X, Y_pred, color='red')
plt.show()