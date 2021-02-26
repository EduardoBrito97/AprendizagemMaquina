import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn_no_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataset_reader import get_dataset, number_of_attributes

dataset = get_dataset(0)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, number_of_attributes[0]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn_no_weight.k = 10
knn_no_weight.train(X_train, y_train)
matrix, report = knn_no_weight.predict(X_test, y_test)