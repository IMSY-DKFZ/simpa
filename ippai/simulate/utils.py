import numpy as np


def randomize(lower_bound, upper_bound, distribution='uniform', size=(1,)):
    mean = (upper_bound + lower_bound) / 2
    spread = (upper_bound - lower_bound)

    if distribution == 'normal':
        return np.random.normal(mean, (spread / 2), size=size)

    if distribution == 'uniform':
        return (np.random.random(size=size) * spread) + lower_bound

    return (upper_bound + lower_bound) / 2

