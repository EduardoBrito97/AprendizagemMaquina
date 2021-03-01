import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

k = 5
classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')

def train(x_train, y_train):
    global classifier
    classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
    classifier.fit(x_train, y_train)

def predict(x_test, y_test):
    y_pred = classifier.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    true_key = 'empty'
    for key in report:
            if 'true' in key.lower():
                    true_key = key
                    break
    if true_key == 'empty':
        report['positives precision'] = 0.0
    else:
        report['positives precision'] = report[true_key]['precision']
    return confusion_matrix(y_test, y_pred, labels=np.unique(y_test)),  report