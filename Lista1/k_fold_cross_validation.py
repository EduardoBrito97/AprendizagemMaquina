import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn_no_weight
import knn_weight
import math
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataset_reader import get_dataset, number_of_attributes

no_weight = "No Weight"
weighted = "Weigthed"

k_fold = 10

dataset = get_dataset(1)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, number_of_attributes[1]].values

total_values = len(x)
values_per_step = math.ceil(len(x) / k_fold)

reports = {}
reports[no_weight] = {}
reports[no_weight][5] = []

matrices = {}
matrices[no_weight] = {}
matrices[no_weight][5] = []

knn_no_weight.k = 5
for i in range(0, len(x), values_per_step):
    last_index = min(len(x), i + values_per_step)
    x_test = x[i:last_index]
    y_test = y[i:last_index]

    x_train = np.concatenate((x[0 : i], x[last_index:len(x)]))
    y_train = np.concatenate((y[0 : i], y[last_index:len(y)])) 

    knn_no_weight.train(x_train, y_train)
    matrix, report = knn_no_weight.predict(x_test, y_test)
    reports[no_weight][knn_no_weight.k].append(report)
    matrices[no_weight][knn_no_weight.k].append(matrix)
    #print(report)

reports[weighted] = {}
reports[weighted][5] = []

matrices[weighted] = {}
matrices[weighted][5] = []
for i in range(0, len(x), values_per_step):
    last_index = min(len(x), i + values_per_step)
    x_test = x[i:last_index]
    y_test = y[i:last_index]

    x_train = np.concatenate((x[0 : i], x[last_index:len(x)]))
    y_train = np.concatenate((y[0 : i], y[last_index:len(y)])) 

    knn_weight.train(x_train, y_train)
    matrix, report = knn_weight.predict(x_test, y_test)
    reports[weighted][knn_no_weight.k].append(report)
    matrices[weighted][knn_no_weight.k].append(matrix)
    #print(report)

reports_medium = {}
for knn_alg in reports.keys():
    reports_medium[knn_alg] = {}
    for k in reports[knn_alg]:
        report_sum = {}
        for current_dict in reports[knn_alg][k]:
            if len(report_sum) == 0:
                report_sum = current_dict
            else:
                for param in current_dict.keys():
                    if type(report_sum[param]) == type({}):
                        for score in report_sum[param]:
                            report_sum[param][score] = report_sum[param][score] + current_dict[param][score]
                    else:
                        report_sum[param] = report_sum[param] + current_dict[param]
        
        for param in report_sum.keys():
                if type(report_sum[param]) == type({}):
                    for score in report_sum[param]:
                        report_sum[param][score] = report_sum[param][score] / k_fold
                else:
                    report_sum[param] = report_sum[param] / k_fold

        reports_medium[knn_alg][k] = report_sum

for knn_alg in reports_medium.keys():
    for k in reports_medium[knn_alg]:
        current_dict = reports_medium[knn_alg][k]
        file_name = str(knn_alg) + "_" + str(k) + "_results.csv"
        with open(file_name, 'w+') as dictwriter:
            fieldnames = list(current_dict.keys())
            writer = csv.DictWriter(dictwriter, fieldnames=fieldnames)
            writer.writeheader()

            for row in current_dict.values():
                writer.writerow(row)