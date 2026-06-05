from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
#x,y = load_digits(return_X_y=True)
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


{X_train, y_train}, {X_test, y_test} = mnist.load_data()
X_train = X_train.reshape(-1, 28*28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28*28).astype(np.float32) / 255.0
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
mlp = MLPClassifier(hidden_layer_sizes=(64,64), activation='relu', solver='adam', max_iter=100)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))