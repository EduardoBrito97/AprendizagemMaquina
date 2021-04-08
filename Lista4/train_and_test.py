import numpy as np
import pandas as pd
import math
import os
import time
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from dataset_reader import get_dataset, number_of_attributes

def test(x_train, y_train, x_test, y_test, best_k):
    y_pred = []
    y_true = []

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(len(x_test)):
        instance_class = 'true' in y_test[index].lower()

        predicted_class = False
        if normalized_dist > class_threshold:
            predicted_class = True

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

def find_best_k(x_train_true, x_train_false, x_train, x_test, y_test):
    best_k = 2
    best_f1_score = 0
    for k in [2, 3, 4, 5, 6]:
        # Criamos KMeans para cada classe, para achar os clusters
        kmeans_true = KMeans(n_clusters=k, random_state=0).fit(x_train_true)
        kmeans_false = KMeans(n_clusters=k, random_state=0).fit(x_train_false)

        # Sabemos agora onde fica cada cluster de cada classe
        true_clusters = kmeans_true.cluster_centers_
        false_clusters = kmeans_false.cluster_centers_
        
        # Precisamos saber agora, no KMeans completo, qual cluster é de qual classe
        full_kmeans = KMeans(n_clusters=k, random_state=0).fit(x_train)
        mapped_clusters = []

        breakpoint()
        # Iterando sobre todos eles, temos como saber à qual classe cada cluster se refere
        for cluster_center in full_kmeans.cluster_centers_:
            if cluster_center in true_clusters:
                mapped_clusters.append(True)
            else:
                mapped_clusters.append(False)

        class_clusters = full_kmeans.predict(x_test)
        y_pred = []

        # Por fim, tentamos predizer qual a classe baseando-se pelo índice do cluster
        for class_cluster in class_clusters:
            y_pred.append(mapped_clusters[class_cluster])

        y_true = []
        # Mapeando agora as classes de verdade para booleano True/False
        for y in y_test:
            y_true.append('true' in y.lower())

        report = classification_report(y_true, y_pred)
        true_f1_score = report[True]['f1-score']
        false_f1_score = report[False]['f1-score']
        avg_f1_score = (true_f1_score + false_f1_score) / 2

        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_k = k

    return best_k
    
def train_and_test(x_train, y_train, x_test, y_test):
    rows_true = []
    rows_false = []
    for i, row in y_train.items():
        if 'true' in str(row).lower():
            rows_true.append(i)
        else:
            rows_false.append(i)

    x_train_true = x_train.iloc[rows_true]
    x_train_false = x_train.iloc[rows_false]
    
    k = find_best_k(x_train_true, x_train_false, x_train, x_test, y_test)

    return test(x_train, y_train, x_test, y_test, k)

def gen_reports_and_statistics(dataset, dataset_target, dataset_name, dataset_index, reports, statistics):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    fold_num = 0
    for train_index, test_index in kfold.split(dataset, dataset_target):
        train = pc1.loc[train_index,:]
        x_train = train.iloc[:, :-1]
        y_train = train.iloc[:, number_of_attributes[dataset_index]]

        test = pc1.loc[test_index,:]
        x_test = test.iloc[:, :-1]
        y_test = test.iloc[:, number_of_attributes[dataset_index]]

        tp, fp, tn, fn, y_pred, y_true = train_and_test(x_train, y_train, x_test, y_test)
        report = classification_report(y_true, y_pred)
        reports['dataset_name'][fold_num] = report
        
        statistics['dataset_name'][fold_num] = {}
        statistics['dataset_name'][fold_num]['tp'] = tp
        statistics['dataset_name'][fold_num]['fp'] = tp
        statistics['dataset_name'][fold_num]['tn'] = tn
        statistics['dataset_name'][fold_num]['fn'] = fn
        fold_num = fold_num + 1

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

    pc1 = get_dataset(0)
    pc1_target = pc1.loc[:, 'defects']

    kc1 = get_dataset(1)
    kc1_target = kc1.loc[:, 'DL']

    gen_reports_and_statistics(pc1, pc1_target, 'pc1', 0, reports, statistics)
    gen_reports_and_statistics(kc1, kc1_target, 'kc1', 1, reports, statistics)
    gen_txt(reports, statistics)