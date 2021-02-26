import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn_no_weight
import knn_weight
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataset_reader import get_dataset, number_of_attributes

k_fold = 10

dataset = get_dataset(1)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, number_of_attributes[1]].values

total_values = len(x)
values_per_step = math.ceil(len(x) / k_fold)

knn_no_weight.k = 5
for i in range(0, len(x), values_per_step):
    last_index = min(len(x), i + values_per_step)
    x_test = x[i:last_index]
    y_test = y[i:last_index]

    x_train = np.concatenate((x[0 : i], x[last_index:len(x)]))
    y_train = np.concatenate((y[0 : i], y[last_index:len(y)])) 

    knn_no_weight.train(x_train, y_train)
    matrix, report = knn_no_weight.predict(x_test, y_test)
    print(report)

print('---------------------------------------------------------------')

knn_weight.k = 5
for i in range(0, len(x), values_per_step):
    last_index = min(len(x), i + values_per_step)
    x_test = x[i:last_index]
    y_test = y[i:last_index]

    x_train = np.concatenate((x[0 : i], x[last_index:len(x)]))
    y_train = np.concatenate((y[0 : i], y[last_index:len(y)])) 

    knn_weight.train(x_train, y_train)
    matrix, report = knn_weight.predict(x_test, y_test)
    print(report)