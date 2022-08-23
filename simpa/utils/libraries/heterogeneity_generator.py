import numpy as np
from sklearn.datasets import make_blobs
from scipy.ndimage.filters import gaussian_filter


class HeterogeneityGeneratorBase(object):
    """
    This is the base class to define heterogeneous structure maps.
    """

    def __init__(self, xdim, ydim, zdim, spacing_mm, target_mean=None,
                 target_std=None, target_min=None, target_max=None,
                 eps=1e-5):
        """
        :param xdim: the x dimension of the volume in voxels
        :param ydim: the y dimension of the volume in voxels
        :param zdim: the z dimension of the volume in voxels
        :param spacing_mm: the spacing of the volume in mm
        :param target_mean: (optional) the mean of the created heterogeneity map
        :param target_std: (optional) the standard deviation of the created heterogeneity map
        :param target_min: (optional) the minimum of the created heterogeneity map
        :param target_max: (optional) the maximum of the created heterogeneity map
        :param eps: (optional) the threshold when a re-normalisation should be triggered (default: 1e-5)
        """
        self._xdim = xdim
        self._ydim = ydim
        self._zdim = zdim
        self._spacing_mm = spacing_mm
        self._mean = target_mean
        self._std = target_std
        self._min = target_min
        self._max = target_max
        self.eps = eps

        self.map = np.ones((self._xdim, self._ydim, self._zdim), dtype=float)

    def get_map(self):
        self.normalise_map()
        return self.map.astype(float)

    def normalise_map(self):
        """
        If mean and std are set, then the data will be normalised to have the desired mean and the
        desired standard deviation.
        If min and max are set, then the data will be normalised to have the desired minimum and the
        desired maximum value.
        If all four values are set, then the data will be normalised to have the desired mean and the
        desired standard deviation first. afterwards all values smaller than min will be ste to min and
        all values larger than max will be set to max.
        """
        # Testing mean mean/std normalisation needs to be done
        if self._mean is not None and self._std is not None:
            if (np.abs(np.mean(self.map) - self._mean) > self.eps or
               np.abs(np.std(self.map) - self._std) > self.eps):
                mean = np.mean(self.map)
                std = np.std(self.map)
                self.map = (self.map - mean) / std
                self.map = (self.map * self._std) + self._mean
            if self._min is not None and self._max is not None:
                self.map[self.map < self._min] = self._min
                self.map[self.map > self._max] = self._max

        # Testing if min max normalisation needs to be done
        if self._min is None or self._max is None:
            return

        if (np.abs(np.min(self.map) - self._min) < self.eps and
           np.abs(np.max(self.map) - self._max) < self.eps):
            return

        _min = np.min(self.map)
        _max = np.max(self.map)
        self.map = (self.map - _min) / (_max-_min)
        self.map = (self.map * (self._max - self._min)) + self._min


class RandomHeterogeneity(HeterogeneityGeneratorBase):
    """
    This heterogeneity generator representes a uniform random sampling between the given bounds.
    Optionally, a Gaussian blur can be specified. Please not that a Gaussian blur will transform the random
    distribution to a Gaussian.
    """

    def __init__(self, xdim, ydim, zdim, spacing_mm, gaussian_blur_size_mm=None, target_mean=None, target_std=None,
                 target_min=None, target_max=None, eps=1e-5):
        """
        :param xdim: the x dimension of the volume in voxels
        :param ydim: the y dimension of the volume in voxels
        :param zdim: the z dimension of the volume in voxels
        :param spacing_mm: the spacing of the volume in mm
        :param gaussian_blur_size_mm: the size of the standard deviation for the Gaussian blur
        :param target_mean: (optional) the mean of the created heterogeneity map
        :param target_std: (optional) the standard deviation of the created heterogeneity map
        :param target_min: (optional) the minimum of the created heterogeneity map
        :param target_max: (optional) the maximum of the created heterogeneity map
        :param eps: (optional) the threshold when a re-normalisation should be triggered (default: 1e-5)
        """
        super().__init__(xdim, ydim, zdim, spacing_mm, target_mean, target_std, target_min, target_max, eps)

        self.map = np.random.random((xdim, ydim, zdim))
        if gaussian_blur_size_mm is not None:
            _gaussian_blur_size_voxels = gaussian_blur_size_mm / spacing_mm
            self.map = gaussian_filter(self.map, _gaussian_blur_size_voxels)


class BlobHeterogeneity(HeterogeneityGeneratorBase):
    """
    This heterogeneity generator representes a blob-like random sampling between the given bounds using the
    sklearn.datasets.make_blobs method. Please look into their documentation for optimising the given hyperparameters.

    """

    def __init__(self, xdim, ydim, zdim, spacing_mm, num_centers=None, cluster_std=None, target_mean=None,
                 target_std=None, target_min=None, target_max=None, random_state=None):
        """
        :param xdim: the x dimension of the volume in voxels
        :param ydim: the y dimension of the volume in voxels
        :param zdim: the z dimension of the volume in voxels
        :param spacing_mm: the spacing of the volume in mm
        :param num_centers: the number of blobs
        :param cluster_std: the size of the blobs
        :param target_mean: (optional) the mean of the created heterogeneity map
        :param target_std: (optional) the standard deviation of the created heterogeneity map
        :param target_min: (optional) the minimum of the created heterogeneity map
        :param target_max: (optional) the maximum of the created heterogeneity map
        :param eps: (optional) the threshold when a re-normalisation should be triggered (default: 1e-5)
        """
        super().__init__(xdim, ydim, zdim, spacing_mm, target_mean, target_std, target_min, target_max)

        if num_centers is None:
            num_centers = int(np.round(np.float_power((xdim * ydim * zdim) * spacing_mm, 1 / 3)))

        if cluster_std is None:
            cluster_std = 1
        x, y = make_blobs(n_samples=(xdim * ydim * zdim) * 10, n_features=3, centers=num_centers, random_state=random_state, cluster_std=cluster_std)

        self.map = np.histogramdd(x, bins=(xdim, ydim, zdim), range=((np.percentile(x[:, 0], 5),
                                                                      np.percentile(x[:, 0], 95)),
                                                                     (np.percentile(x[:, 1], 5),
                                                                         np.percentile(x[:, 1], 95)),
                                                                     (np.percentile(x[:, 2], 5),
                                                                         np.percentile(x[:, 2], 95))))[0]
        self.map = gaussian_filter(self.map, 5)
