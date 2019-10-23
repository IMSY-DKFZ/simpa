import numpy as np
from scipy.ndimage import gaussian_filter


def randomize(lower_bound, upper_bound, distribution='uniform', size=(1,), gauss_kernel_size=0):
    """
    This function creates a randomly distributed variable between lower_bound and upper_bound.

    The number is drawn from obe of the following distributions:
        - 'uniform'
        - 'normal'

    The default behaviour is to draw from a uniform distribution.

    If a gauss_kernel_size > 0 is given, then a gaussian filter is applied to the random distribution.

    :param lower_bound: the lower bound of the random distribution
    :param upper_bound: the upper bound of the random distribution
    :param distribution:  string argument: 'uniform', 'normal'
    :param size: target shape of the output array
    :param gauss_kernel_size: if >0 then a filter is applied to the distribution with filter size gauss_kernel_size.

    :return: a size shaped array of random numbers
    """
    mean = (upper_bound + lower_bound) / 2
    spread = (upper_bound - lower_bound)

    if distribution == 'normal':
        result = np.random.normal(mean, (spread / 2), size=size)
    elif distribution == 'uniform':
        result = (np.random.random(size=size) * spread) + lower_bound
    else:
        result = (np.random.random(size=size) * spread) + lower_bound

    if gauss_kernel_size > 0:
        result = gaussian_filter(result, gauss_kernel_size)
        # After filtering, the noise needs to be rescaled to be in the correct range again (esp. wrt the spread)
        result = ((result - np.mean(result)) / np.std(result)) * \
                 (spread / 2) + mean

    return result


def gruneisen_parameter_from_temperature(temperature_in_celcius):
    """
    This function returns the dimensionless gruneisen parameter based on a heuristic formular that
    was determined experimentally:

    @book{wang2012biomedical,
        title={Biomedical optics: principles and imaging},
        author={Wang, Lihong V and Wu, Hsin-i},
        year={2012},
        publisher={John Wiley \& Sons}
    }

    :param temperature_in_celcius: the temperature in degrees celcius
    :return: a floating point number, if temperature_in_celcius is a number or a
    float array, if temperature_in_celcius is an array
    """

    return 0.0043 + 0.0053 * temperature_in_celcius