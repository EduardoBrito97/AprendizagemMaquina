import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Pegamos os novos conjuntos gerados pelo kmeans, treinamos o Gaussian NB e predizemos o x_teste
def test(x_train, y_train, x_test, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train.values.ravel(order='C')).predict(x_test)
    y_true = []
    for index in range(len(x_test)):
        instance_class = 'true' in y_test.iloc[index].values[0].lower()
        predicted_class = y_pred[index]
        if predicted_class == True and instance_class == True:
            tp = tp + 1
        elif predicted_class == True and instance_class == False:
            fp = fp + 1
        elif predicted_class == False and instance_class == False:
            tn = tn + 1
        else:
            fn = fn + 1

        y_true.append(instance_class)
    return tp, fp, tn, fn, y_pred, np.array(y_true)

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

    # Separando o x de teste entre x_true e x_false
    for _, row in y_train.iterrows():
        if 'true' in str(row).lower():
            rows_true.append(index)
        else:
            rows_false.append(index)
        index = index + 1

    # Para cada classe, procuramos o melhor K usando elbow/silhouette
    x_train_true = x_train.iloc[rows_true]
    k_true = find_best_k(x_train_true)

    x_train_false = x_train.iloc[rows_false]
    k_false = find_best_k(x_train_false)

    kmeans_true = KMeans(n_clusters=k_true).fit(x_train_true)
    kmeans_false = KMeans(n_clusters=k_false).fit(x_train_false)

    # Criamos grupos com os clusters pra x e valores corretos para y
    new_x_train = []
    new_y_train = []

    for cluster_center in kmeans_true.cluster_centers_:
        current_dict_x = {}
        current_dict_y = {}

        index = 0
        for column in x_train.columns:
            current_dict_x[column] = cluster_center[index]
            index = index + 1
        
        for column in y_train.columns:
            current_dict_y[column] = True

        new_x_train.append(current_dict_x)
        new_y_train.append(current_dict_y)

    for cluster_center in kmeans_false.cluster_centers_:
        current_dict_x = {}
        current_dict_y = {}

        index = 0
        for column in x_train.columns:
            current_dict_x[column] = cluster_center[index]
            index = index + 1
        
        for column in y_train.columns:
            current_dict_y[column] = False

        new_x_train.append(current_dict_x)
        new_y_train.append(current_dict_y)

    pandas_x_train = pd.DataFrame.from_dict(new_x_train)
    pandas_y_train = pd.DataFrame.from_dict(new_y_train)

    # Por fim, testamos usando Gaussian Nayve Bayes usando os novos grupos
    return test(pandas_x_train, pandas_y_train, x_test, y_test)
