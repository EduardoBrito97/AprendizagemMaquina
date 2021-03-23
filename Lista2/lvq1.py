from sklearn.neighbors import KNeighborsClassifier
from prototype_gen import random_prototypes
from lvq_base import update_prototype, num_of_prot_training_epochs

weight = 0.25

def gen_prototypes(x_train, y_train):
    # Gerando os protótipos para poder fazer o prodecedimento do LVQ1
    prototypes_x, prototypes_y, x_train, y_train = random_prototypes(x_train, y_train)

    classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    classifier.fit(prototypes_x, prototypes_y)

    for _ in range(num_of_prot_training_epochs):
        predicted_classes = classifier.predict(x_train)
        real_instances = classifier.kneighbors(X=x_train, n_neighbors=1, return_distance=False)
        for i in range(len(x_train)):
            real_instance = x_train[i]
            real_instance_class = y_train[i]

            # Descobrindo qual seria a classe do protótipo ao usar knn pra classificá-lo
            prototype_class = predicted_classes[i]
            prototype_index = real_instances[i]
            prototype = prototypes_x[prototype_index][0]

            is_same_class = prototype_class == real_instance_class
            
            # Atualizando o protótipo baseado nas classes e na distância
            update_prototype(prototype, real_instance, is_same_class, weight)

    return prototypes_x, prototypes_y