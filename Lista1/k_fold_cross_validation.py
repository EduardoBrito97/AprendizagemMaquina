import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn_no_weight
import knn_weight
import adaptative_knn
import math
import json
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataset_reader import get_dataset, number_of_attributes, datasets

no_weight = "No Weight"
weighted = "Weigthed"
adaptative = "Adaptative"

k_fold = 10

reports = {}
matrices = {}

ks = [1, 2, 3, 5, 7, 11, 13, 15]

def train_and_get_reports(algorithm, dataset_index, trainer):
    dataset = get_dataset(dataset_index)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, number_of_attributes[dataset_index]].values

    values_per_step = math.ceil(len(x) / k_fold)

    reports[algorithm] = {}
    matrices[algorithm] = {}
    for k in ks:
        trainer.k = k
        reports[algorithm][k] = []
        matrices[algorithm][k] = []

        for i in range(0, len(x), values_per_step):
            last_index = min(len(x), i + values_per_step)
            x_test = x[i:last_index]
            y_test = y[i:last_index]

            x_train = np.concatenate((x[0 : i], x[last_index:len(x)]))
            y_train = np.concatenate((y[0 : i], y[last_index:len(y)])) 

            start_time = time.time()
            trainer.train(x_train, y_train)
            train_time = time.time() - start_time

            start_time = time.time()
            matrix, report = trainer.predict(x_test, y_test)
            test_time = time.time() - start_time

            report['train_time_in_ms'] = int(train_time * 1000)
            report['test_time_in_ms'] = int(test_time * 1000)
            reports[algorithm][k].append(report)
            matrices[algorithm][k].append(matrix)
            #print(report)

def sum_reports(report_sum, current_dict):
    for param in current_dict.keys():
        if type(report_sum[param]) == type({}):
            for score in report_sum[param]:
                report_sum[param][score] = report_sum[param][score] + current_dict[param][score]
        else:
            report_sum[param] = report_sum[param] + current_dict[param]

def get_reports_avg(report_sum):
    for param in report_sum.keys():
        if type(report_sum[param]) == type({}):
            for score in report_sum[param]:
                report_sum[param][score] = report_sum[param][score] / k_fold
        else:
            report_sum[param] = report_sum[param] / k_fold
    return report_sum

def process_reports():
    reports_avg = {}
    for knn_alg in reports.keys():
        reports_avg[knn_alg] = {}
        for k in reports[knn_alg]:
            report_sum = {}
            for current_dict in reports[knn_alg][k]:
                if len(report_sum) == 0:
                    report_sum = current_dict
                else:
                    sum_reports(report_sum, current_dict)
            
            report_sum = get_reports_avg(report_sum)

            reports_avg[knn_alg][k] = report_sum
    return reports_avg

def train_and_test_on_dataset_and_save_results(dataset_index):
    dataset_name = datasets[dataset_index]
    train_and_get_reports(no_weight, dataset_index, knn_no_weight)
    train_and_get_reports(weighted, dataset_index, knn_weight)
    train_and_get_reports(adaptative, dataset_index, adaptative_knn)
    reports_avg = process_reports()

    folder_name = "results/" + dataset_name + "/" 
    #if os.path.isfile(folder_name) == False:
    #    os.mkdir(folder_name, mode = 0o666)
    for knn_alg in reports_avg.keys():
        for k in reports_avg[knn_alg]:
            current_dict = reports_avg[knn_alg][k]
            file_name = folder_name + str(knn_alg) + "_" + str(k) + "_results.txt"
            fo = open(file_name, "w")
            for k, v in current_dict.items():
                fo.write(str(k) + ' = '+ str(v) + '\n')
            fo.close()

if __name__ == '__main__':
    train_and_test_on_dataset_and_save_results(0)
    train_and_test_on_dataset_and_save_results(1)
