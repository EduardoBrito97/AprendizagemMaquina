from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

k = 5
classifier = KNeighborsClassifier(n_neighbors=k)

def train(x_train, y_train):
    classifier.fit(x_train, y_train)

def predict(x_test, y_test):
    y_pred = classifier.predict(x_test)
    return confusion_matrix(y_test, y_pred),  classification_report(y_test, y_pred)