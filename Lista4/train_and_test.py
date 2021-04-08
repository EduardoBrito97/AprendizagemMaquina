import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
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

def select_using_elbow_method(distortions):
    for index in range(len(distortions)):
        if index == 0 or index == (len(distortions) - 1):
            continue
        
        previous_distortion = distortions[index - 1]
        curr_distortion = distortions[index]
        next_distortion = distortions[index + 1]

        diff_from_previous = previous_distortion - curr_distortion
        diff_from_next = curr_distortion - next_distortion
        
        # Procuramos uma quebra muito grande nas diferenças entre o anterior e o atual e o atual e o posterior,
        # onde 2 foi um dado experimental
        if diff_from_previous > (diff_from_next * 2):
            return index, True

    return index, False

# Primeiramente tenta-se usar o método do "Elbow";
# Caso falhe, usamos a método "Silhouette"
def find_best_k(x_train):
    ks = [2, 3, 4, 5, 6]

    distortions = []
    silhouettes = []
    for k in ks:
        kmeans = KMeans(n_clusters=k).fit(x_train)
        
        # Por via das dúvidas, já armazenamos o score da silhouette
        labels = kmeans.labels_
        silhouettes.append(silhouette_score(x_train, labels, metric = 'euclidean'))

        # E calculamos o erro quadrático médio de cada cluster para fazer o método do elbow
        distortions.append(kmeans.inertia_)

    # Por fim, decidimos qual método usar
    index, elbow_found = select_using_elbow_method(distortions)    
    if not elbow_found:
        index = silhouettes.index(max(silhouettes))
    return ks[index]
    
def train_and_test(x_train, y_train, x_test, y_test):
    rows_true = []
    rows_false = []
    index = 0
    for _, row in y_train.items():
        if 'true' in str(row).lower():
            rows_true.append(index)
        else:
            rows_false.append(index)
        index = index + 1
    
    x_train_true = x_train.iloc[rows_true]
    k_true = find_best_k(x_train_true)

    x_train_false = x_train.iloc[rows_false]
    k_false = find_best_k(x_train_false)

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