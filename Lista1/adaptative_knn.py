import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

k = 5

x_training = []
y_training = []
adaptive_v = []

def dist(alfa, beta):
    return np.sqrt(np.sum((alfa-beta)**2))

def train(x_train, y_train):
    global x_training, y_training, adaptive_v
    x_training = x_train
    y_training = y_train

    adaptive_v = np.zeros([len(x_train)]) 
    for i in range(x_train.shape[0]):
        min_rad = 999999
        for j in range(x_train.shape[0]):
            if i == j or y_train[i] == y_train[j]:
                continue
            rad = dist(x_train[i], x_train[j])
            if rad < min_rad:
                min_rad = rad
        adaptive_v[i] = min_rad * 0.99999

def get_pred_class(dist):
        dist = dist / (adaptive_v + 1e-5)
        min_index = dist.argsort()[:k]
        cand_class = y_training[min_index]
        unique, freq = np.unique(cand_class, return_counts=True)
        return unique[freq.argmax()]
            
def predict_one(test_instance):
    dists = np.empty(len(y_training))
    for idx, instance in enumerate(x_training):
        dists[idx] = dist(test_instance, instance)

    return get_pred_class(dists)

def contains_ignore_case(source, verific):
    return verific.lower() in source.lower()

def predict(x_test, y_test):
    pred = []
    for test_instance in x_test:
        pred.append(predict_one(test_instance))
    
    i = 0
    total_correct = int(0)
    true_positives = int(0)
    true_negatives = int(0)
    false_positives = int(0)
    false_negatives = int(0)
    for y in y_test:
        predicted = str(pred[i])
        correct = str(y)
        if correct == predicted:
            total_correct = total_correct + 1

        if contains_ignore_case(predicted, "true") and contains_ignore_case(correct, "true"):
            true_positives = true_positives + 1

        if contains_ignore_case(predicted, "true") and contains_ignore_case(correct, "false"):
            false_positives = false_positives + 1

        if contains_ignore_case(predicted, "false") and contains_ignore_case(correct, "false"):
            true_negatives = true_negatives + 1

        if contains_ignore_case(predicted, "false") and contains_ignore_case(correct, "true"):
            false_negatives = false_negatives + 1

        i = i + 1
    
    positives_precision = 0 
    if true_positives + false_negatives != 0:
        positives_precision = true_positives / (true_positives + false_negatives)
    
    negatives_precision = 0
    if true_negatives + false_positives != 0:
        negatives_precision = true_negatives / (true_negatives + false_positives)

    report = {}
    report['accuracy'] = total_correct/len(y_test)
    report['true positives'] = true_positives
    report['true negatives'] = true_negatives
    report['false positives'] = false_positives
    report['false negatives'] = false_negatives
    report['positives precision'] = positives_precision
    report['negatives precision'] = negatives_precision

    matrix = str(true_positives) + " " + str(false_positives) + "\n" + str(true_negatives) + " " + str(false_negatives)

    return matrix, report