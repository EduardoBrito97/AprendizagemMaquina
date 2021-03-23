import numpy as np
import lvq1
from sklearn.neighbors import KNeighborsClassifier
from prototype_gen import random_prototypes
from lvq_base import update_prototype, num_of_prot_training_epochs

weight = 0.25

def gen_prototypes(x_train, y_train):
    # Gerando os protótipos para poder fazer o prodecedimento do LVQ2.1
    prototypes_x, prototypes_y = lvq1.gen_prototypes(x_train, y_train)

    classifier = KNeighborsClassifier(n_neighbors=2, weights='uniform')
    classifier.fit(prototypes_x, prototypes_y)

    for _ in range(num_of_prot_training_epochs):
        x_neighbors = classifier.kneighbors(X=x_train, n_neighbors=2, return_distance=False)
        for i in range(len(x_train)):
            x_instance = x_train[i]
            x_class = y_train[i]

            # Achando quem é o vizinho mais próximo da instância real 
            # em um knn treinado apenas com os protótipos e também a distância pra instância
            pi_index = x_neighbors[i][0]
            pi = prototypes_x[pi_index]
            pi_class = prototypes_y[pi_index]
            pi_dist = np.linalg.norm(pi - x_instance)

            # Achando quem é o segundo vizinho mais próximo da instância real 
            # em um knn treinado apenas com os protótipos e também a distância pra instância
            pj_index = x_neighbors[i][1]
            pj = prototypes_x[pj_index]
            pj_class = prototypes_y[pj_index]
            pj_dist = np.linalg.norm(pj - x_instance)

            # Fazendo cálculo da janela para saber se os protótipos serão atualizados
            # 0.0000001 é adicionado para evitar divisão por 0
            distpipj = pi_dist / (pj_dist + 0.0000001)
            distpjpi = pj_dist / (pi_dist + 0.0000001)
            dist = min(distpipj, distpjpi)
            is_inside_window = dist > ((1 - weight) / (1 + weight))

            # Atualizando os protótipos baseados na distância, caso sejam de diferentes classes
            if is_inside_window and pi_class != pj_class:
                update_prototype(pi, x_instance, pi_class == x_class, weight)
                update_prototype(pj, x_instance, pj_class == x_class, weight)

    return prototypes_x, prototypes_y