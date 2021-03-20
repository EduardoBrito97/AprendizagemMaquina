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

    classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    classifier.fit(x_train, y_train)

    for _ in range(100):
        predicted_classes = classifier.predict(prototypes_x)
        real_instances = classifier.kneighbors(X=prototypes_x, n_neighbors=1, return_distance=False)
        for i in range(len(prototypes_x)):
            prototype = prototypes_x[i]
            prototype_class = prototypes_y[i]

            predicted_class = predicted_classes[i]
            real_instance_index = real_instances[i]
            real_instance = x_train[real_instance_index][0]

            is_same_class = prototype_class == predicted_class
            update_prototype(prototype, real_instance, is_same_class)

    return prototypes_x, prototypes_y