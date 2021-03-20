import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def random_prototypes(x, y):
    # Geraremos 10% do conjunto de treinamento como protótipos
    num_of_prototypes = int(len(x)/10)
    prototypes_x = []
    prototypes_y = []

    # Como não é usado o K-Fold estratificado e existem datasets muito desbalanceados,
    # acontece de um padrão de treinamento não possuir ambas as classes, então tentar garantir
    # ao menos um protótipo de cada classe baseado no set de treinamento pode ser impossível
    for _ in range(num_of_prototypes):
        index = random.randint(0, len(x)-1)
        
        prototype_x = x[index]
        prototype_y = y[index]
        
        prototypes_x.append(prototype_x)
        prototypes_y.append(prototype_y)
        
        # Removendo os protótipos da lista de treinamento, pois ele será usado para procurar os vizinhos
        # mais pra frente. Caso eles mesmos ainda estejam na lista de treinamento, a distância vai ser 0,
        # logo não será atualizado
        x_list = x.tolist()
        y_list = y.tolist()

        x_list.pop(index)
        y_list.pop(index)

        x = np.array(x_list)
        y = np.array(y_list)

    return np.array(prototypes_x), np.array(prototypes_y), x, y