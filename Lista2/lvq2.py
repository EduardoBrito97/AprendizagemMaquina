import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from prototype_gen import random_prototypes

weight = 0.25
num_of_prototypes = 100

def update_prototype(prototype, real_instance, is_same_class):
    for j in range(len(prototype)):
        prototype_val = prototype[j]
        real_instance_val = real_instance[j]
        weighted_diff = weight * (real_instance_val - prototype_val)
        if is_same_class:
            prototype[j] = prototype_val + weighted_diff
        else:
            prototype[j] = prototype_val - weighted_diff

def gen_prototypes(x_train, y_train):
    prototypes_x, prototypes_y = random_prototypes(x_train, y_train, 50)

    classifier = KNeighborsClassifier(n_neighbors=2, weights='uniform')
    classifier.fit(prototypes_x, prototypes_y)

    for _ in range(100):
        x_neighbors = classifier.kneighbors(X=x_train, n_neighbors=2, return_distance=False)
        for i in range(len(x_train)):
            x_instance = x_train[i]
            x_class = y_train[i]

            pi_index = x_neighbors[i][0]
            pi = prototypes_x[pi_index]
            pi_class = prototypes_y[pi_index]
            pi_dist = np.linalg.norm(pi - x_instance)

            pj_index = x_neighbors[i][1]
            pj = prototypes_x[pj_index]
            pj_class = prototypes_y[pj_index]
            pj_dist = np.linalg.norm(pj - x_instance)

            distpipj = pi_dist / (pj_dist + 0.0000001)
            distpjpi = pj_dist / (pi_dist + 0.0000001)
            dist = min(distpipj, distpjpi)
            is_inside_window = dist > ((1 - weight) / (1 + weight))

            if is_inside_window and pi_class != pj_class:
                update_prototype(pi, x_instance, pi_class == x_class)
                update_prototype(pj, x_instance, pj_class == x_class)            

    return prototypes_x, prototypes_y