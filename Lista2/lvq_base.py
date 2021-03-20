num_of_prot_training_epochs = 100

def update_prototype(prototype, real_instance, is_same_class, weight, epslon = 1):
    for j in range(len(prototype)):
        prototype_val = prototype[j]
        real_instance_val = real_instance[j]
        weighted_diff = epslon * weight * (real_instance_val - prototype_val)
        if is_same_class:
            prototype[j] = prototype_val + weighted_diff
        else:
            prototype[j] = prototype_val - weighted_diff