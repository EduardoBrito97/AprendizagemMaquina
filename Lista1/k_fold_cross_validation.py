import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn_no_weight
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataset_reader import get_dataset, number_of_attributes

k_fold = 10

dataset = get_dataset(0)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, number_of_attributes[0]].values

print(type(x))

total_values = len(x)
values_per_step = int(len(x) / k_fold)
knn_no_weight.k = 10
for i in range(0, len(x), values_per_step):
    last_index = min(len(x), i + values_per_step)
    x_test = x[i:last_index]
    y_test = y[i:last_index]

    print(str(i) + ".." + str(last_index))
    print(x_test)
    continue

    x_train = np.concatenate((x[0 : i], x[last_index:len(x)]))
    y_train = np.concatenate((y[0 : i], y[last_index:len(y)])) 

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    knn_no_weight.train(x_train, y_train)
    matrix, report = knn_no_weight.predict(x_test, y_test)
    print(report)