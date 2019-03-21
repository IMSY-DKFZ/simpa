import numpy as np
from scipy.ndimage import gaussian_filter


def randomize(lower_bound, upper_bound, distribution='uniform', size=(1,), gauss_kernel_size=0):
    mean = (upper_bound + lower_bound) / 2
    spread = (upper_bound - lower_bound)

    result = (upper_bound + lower_bound) / 2

    if distribution == 'normal':
        result = np.random.normal(mean, (spread / 2), size=size)

        if gauss_kernel_size > 0:
            result = gaussian_filter(result, gauss_kernel_size)

            # After filtering, the noise needs to be rescaled to be in the correct range again (esp. wrt the spread)
            result = ((result - np.mean(result)) / np.std(result)) * \
                     (spread / 2) + mean

    if distribution == 'uniform':
        result = (np.random.random(size=size) * spread) + lower_bound

    return result

