import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads-1fef94ba-27e1-4fab-a7ad-56867b8fb5a1.csv')
X = dataset.iloc[:, [1, 2]].values
Y = dataset.iloc[:, 3].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.25, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=Y_train, palette={0: 'red', 1: 'blue'}, marker ='o')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Coffecients:", classifier.coef_)
print("Intercept:", classifier.intercept_)


#import seaborn as 
x1 = np.linspace(-10,10,100)
x2 = (-2.0766837* x1 + 0.95217247)/1.11008221
plt.figure(figsize=(8, 6))
plt.plot(x1, x2, color='green', label='Decision Boundary')
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=Y_train, palette={0: 'red', 1: 'blue'}, marker ='o')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()



from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, Y_train)
y_pred_knn = knn_classifier.predict(X_test)
print("KNN Accuracy:", accuracy_score(Y_test, y_pred_knn))


x1_vals = np.linspace(-3, 3, 400)
x2_vals = np.linspace(-3, 3, 400)
x1,x2 = np.meshgrid(x1_vals, x2_vals)
Z = knn_classifier.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)
print(Z)
