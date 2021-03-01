import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from knn_weight import true_positive_precision

k = 5
classifier = KNeighborsClassifier(n_neighbors=k, weights='uniform')

def train(x_train, y_train):
    global classifier
    classifier = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    classifier.fit(x_train, y_train)

def predict(x_test, y_test):
    y_pred = classifier.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    true_key = 'empty'
    for key in report:
            if 'true' in key.lower():
                    true_key = key
                    break
    # Isso acontece quando nenhuma das instâncias foram classificadas como verdadeira
    if true_key == 'empty':
        # Caso não haja nenhuma classe verdadeira realmente, o true_positive é 100%
        report[true_positive_precision] = 1.0
        for key in y_test:
            if 'true' in key.lower():
                report[true_positive_precision] = 0.0 # 0%, caso contrário
    else:
        report[true_positive_precision] = report[true_key]['precision']
    return confusion_matrix(y_test, y_pred, labels=np.unique(y_test)),  report