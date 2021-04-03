import numpy as np
import pandas as pd
import math
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataset_reader import get_1_class_instance, number_of_attributes

filter_threshold = 0.15
class_threshold = 0.05

def filter_instances(x, y):
    classifier = KNeighborsClassifier(n_neighbors=2, weights='uniform')
    classifier.fit(x, y)
    x_neighbors = classifier.kneighbors(X=x, n_neighbors=2, return_distance=False)

    num_of_attr = len(x[0])
    min_inst = np.zeros(num_of_attr)
    max_inst = np.ones(num_of_attr)
    max_dist = np.linalg.norm(max_inst - min_inst)

    removed_instances = []
    for index in range(len(x)):
        # Achando quem é o segundo vizinho mais próximo da instância 
        # uma vez que o primeiro vizinho mais próximo é a própria instância
        neighbor_index = x_neighbors[index][1]
        neighbor = x[neighbor_index]
        neighbor_dist = np.linalg.norm(x[index] - neighbor)/max_dist
        if neighbor_dist > filter_threshold:
            removed_instances.append(index)

    new_x = np.delete(x, removed_instances, axis=0)
    new_y = y[0:(len(y) - len(removed_instances))]

    return new_x, new_y

def test(x_train, y_train, x_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    classifier.fit(x_train, y_train)
    x_neighbors = classifier.kneighbors(X=x_test, n_neighbors=1, return_distance=False)

    y_pred = []
    y_true = []

    num_of_attr = len(x_test[0])
    min_inst = np.zeros(num_of_attr)
    max_inst = np.ones(num_of_attr)
    max_dist = np.linalg.norm(max_inst - min_inst)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(len(x_test)):
        instance_class = 'true' in y_test[index].lower()

        neighbor_index = x_neighbors[index][0]
        neighbor = x_train[neighbor_index]

        neighbor_dist = np.linalg.norm(x_test[index] - neighbor) / max_dist
        normalized_dist = neighbor_dist 
        
        predicted_class = True
        if normalized_dist > class_threshold:
            predicted_class = False

        if predicted_class == True and instance_class == True:
            tp = tp + 1
        elif predicted_class == True and instance_class == False:
            fp = fp + 1
        elif predicted_class == False and instance_class == False:
            tn = tn + 1
        else:
            fn = fn + 1

        y_pred.append(predicted_class)
        y_true.append(instance_class)

    return tp, fp, tn, fn, y_pred, y_true
    
def train_and_test(dataset_index):
    dataset_true = get_1_class_instance(dataset_index, True)
    dataset_false = get_1_class_instance(dataset_index, False)

    x_true = dataset_true.iloc[:, :-1].values
    y_true = dataset_true.iloc[:, number_of_attributes[dataset_index]].values

    x_false = dataset_false.iloc[:, :-1].values
    y_false = dataset_false.iloc[:, number_of_attributes[dataset_index]].values

    num_of_training_classes = math.ceil(( len(x_false) * train_class_percentage) / 100)

    x_train = x_false[0:num_of_training_classes]
    y_train = y_false[0:num_of_training_classes]

    x_test = np.concatenate((x_false[num_of_training_classes : len(x_false)], x_true))
    y_test = np.concatenate((y_false[num_of_training_classes : len(y_false)], y_true))

    x_train, y_train = filter_instances(x_train, y_train)

    return test(x_train, y_train, x_test, y_test)

def train_and_test_on_dataset_and_save_results(dataset_index):
    dataset_name = datasets[dataset_index]

    folder_name = "results/" + dataset_name + "/" 
    if os.path.exists(folder_name) == False:
            os.mkdir(folder_name, mode = 0o666)
    for knn_alg in reports.keys():
        for k in reports[knn_alg]:
            current_dict = reports[knn_alg][k]
            file_name = folder_name + str(knn_alg) + "_" + str(k) + "_results.txt"
            fo = open(file_name, "w")
            for k, v in current_dict.items():
                fo.write(str(k) + ' = '+ str(v) + '\n')
            fo.close()

def gen_txt(reports, statistics):
    for dataset in reports.keys():
        folder_name = "results/" + dataset + "/" 
        for percentage in reports[dataset]:
            report = reports[dataset][percentage]
            statistic = statistics[dataset][percentage]

            file_name = folder_name + str(percentage) + "_results.txt"
            fo = open(file_name, "w")
            fo.write(report)
            fo.write(str(statistic))

            fo.close()

if __name__ == '__main__':
    global train_class_percentage
    statistics = {}
    statistics['pc1'] = {}
    statistics['kc1'] = {}

    reports = {}
    reports['pc1'] = {}
    reports['kc1'] = {}

    for percentage in [30, 40, 50]:
        train_class_percentage = percentage
        
        tp, fp, tn, fn, y_pred, y_true = train_and_test(0)
        report = classification_report(y_true, y_pred)
        reports['pc1'][percentage] = report
        
        statistics['pc1'][percentage] = {}
        statistics['pc1'][percentage]['tp'] = tp
        statistics['pc1'][percentage]['fp'] = tp
        statistics['pc1'][percentage]['tn'] = tn
        statistics['pc1'][percentage]['fn'] = fn

        tp, fp, tn, fn, y_pred, y_true = train_and_test(1)
        report = classification_report(y_true, y_pred)
        reports['kc1'][percentage] = report
        
        statistics['kc1'][percentage] = {}
        statistics['kc1'][percentage]['tp'] = tp
        statistics['kc1'][percentage]['fp'] = tp
        statistics['kc1'][percentage]['tn'] = tn
        statistics['kc1'][percentage]['fn'] = fn

    gen_txt(reports, statistics)