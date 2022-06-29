import numpy as np
from sklearn.datasets import make_blobs
from scipy.ndimage.filters import gaussian_filter


class HeterogeneityGeneratorBase(object):
    """
    This is the base class to define heterogeneous structure maps.
    """

    def __init__(self, _xdim, _ydim, _zdim, _spacing_mm, _mean=None, _std=None, _min=None, _max=None):
        """
        @param _xdim: the x dimension of the volume in voxels
        @param _ydim: the y dimension of the volume in voxels
        @param _zdim: the z dimension of the volume in voxels
        @param _spacing_mm: the spacing of the volume in mm
        @param _mean: (optional) the mean of the created heterogeneity map
        @param _std: (optional) the standard deviation of the created heterogeneity map
        @param _min: (optional) the minimum of the created heterogeneity map
        @param _max: (optional) the maximum of the created heterogeneity map
        """
        self._xdim = _xdim
        self._ydim = _ydim
        self._zdim = _zdim
        self._spacing_mm = _spacing_mm
        self._mean = _mean
        self._std = _std
        self._min = _min
        self._max = _max
        self.EPS = 1e-5

        self.map = np.ones((self._xdim, self._ydim, self._zdim))

    def get_map(self):
        self.normalise_map()
        return self.map

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
        if self._mean is not None and self._std is not None:
            if (np.abs(np.mean(self.map) - self._mean) > self.EPS or
               np.abs(np.std(self.map) - self._std) > self.EPS):
                mean = np.mean(self.map)
                std = np.std(self.map)
                self.map = (self.map - mean) / std
                self.map = (self.map * self._std) + self._mean
            if self._min is not None and self._max is not None:
                self.map[self.map < self._min] = self._min
                self.map[self.map > self._max] = self._max
        else:
            if self._min is not None and self._max is not None:
                if (np.abs(np.min(self.map) - self._min) > self.EPS or
                   np.abs(np.max(self.map) - self._max) > self.EPS):
                    _min = np.min(self.map)
                    _max = np.max(self.map)
                    self.map = (self.map - _min) / (_max-_min)
                    self.map = (self.map * (self._max - self._min)) + self._min


class RandomHeterogeneity(HeterogeneityGeneratorBase):

    def __init__(self, _xdim, _ydim, _zdim, _spacing_mm, _gaussian_blur_size_mm=None, _mean=None, _std=None, _min=None, _max=None):
        super().__init__(_xdim, _ydim, _zdim, _spacing_mm, _mean, _std, _min, _max)

        self.map = np.random.random((_xdim, _ydim, _zdim))
        if _gaussian_blur_size_mm is not None:
            _gaussian_blur_size_voxels = _gaussian_blur_size_mm / _spacing_mm
            self.map = gaussian_filter(self.map, _gaussian_blur_size_voxels)


class BlobHeterogeneity(HeterogeneityGeneratorBase):

    def __init__(self, _xdim, _ydim, _zdim, _spacing_mm, _centers=None, _cluster_std=None, _mean=None,
                 _std=None, _min=None, _max=None, _random_state=None):
        super().__init__(_xdim, _ydim, _zdim, _spacing_mm, _mean, _std, _min, _max)

        if _centers is None:
            _centers = int(np.round(np.float_power((_xdim * _ydim * _zdim) * _spacing_mm, 1/3)))

        if _cluster_std is None:
            _cluster_std = 1
        x, y = make_blobs(n_samples=(_xdim * _ydim * _zdim) * 10, n_features=3, centers=_centers, random_state=_random_state, cluster_std=_cluster_std)

        self.map = np.histogramdd(x, bins=(_xdim, _ydim, _zdim), range=((np.percentile(x[:, 0], 5),
                                                                         np.percentile(x[:, 0], 95)),
                                                                        (np.percentile(x[:, 1], 5),
                                                                         np.percentile(x[:, 1], 95)),
                                                                        (np.percentile(x[:, 2], 5),
                                                                         np.percentile(x[:, 2], 95))))[0]
        self.map = gaussian_filter(self.map, 5)
