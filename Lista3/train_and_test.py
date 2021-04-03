import numpy as np
import pandas as pd
import math
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataset_reader import get_1_class_instance, number_of_attributes, datasets

reports = {}
matrices = {}

train_class_percentage = 30
filter_threshold = 0.05
class_threshold = 0.03

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
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    num_of_attr = len(x_test[0])
    min_inst = np.zeros(num_of_attr)
    max_inst = np.ones(num_of_attr)
    max_dist = np.linalg.norm(max_inst - min_inst)

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
            true_positives = true_positives + 1
        elif predicted_class == True and instance_class == False:
            false_positives = false_positives + 1
        elif predicted_class == False and instance_class == False:
            true_negatives = true_negatives + 1
        else:
            false_negatives = false_negatives + 1

    return true_positives, false_positives, true_negatives, false_negatives
    
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

if __name__ == '__main__':
    tp, fp, tn, fn = train_and_test(1)
    print('tp ' + str(tp))
    print('fp ' + str(fp))
    print('tn ' + str(tn))
    print('fn ' + str(fn))
