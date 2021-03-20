import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def random_prototypes(x, y, num_of_prototypes):
    prototypes_x = []
    prototypes_y = []

    for _ in range(num_of_prototypes):
        index = random.randint(0, len(x)-1)
        
        prototype_x = x[index]
        prototype_y = y[index]
        
        prototypes_x.append(prototype_x)
        prototypes_y.append(prototype_y)

    return np.array(prototypes_x), np.array(prototypes_y)