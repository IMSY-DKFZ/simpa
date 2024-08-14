# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from sklearn.datasets import make_blobs
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
from simpa.utils import Tags
from typing import Union, Optional
from simpa.log import Logger
import os
import requests
import zipfile


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
    This heterogeneity generator represents a uniform random sampling between the given bounds.
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
        """
        super().__init__(xdim, ydim, zdim, spacing_mm, target_mean, target_std, target_min, target_max)

        if num_centers is None:
            num_centers = int(np.round(np.float_power((xdim * ydim * zdim) * spacing_mm, 1 / 3)))

        if cluster_std is None:
            cluster_std = 1
        x, y = make_blobs(n_samples=(xdim * ydim * zdim) * 10, n_features=3, centers=num_centers,
                          random_state=random_state, cluster_std=cluster_std)

        self.map = np.histogramdd(x, bins=(xdim, ydim, zdim), range=((np.percentile(x[:, 0], 5),
                                                                      np.percentile(x[:, 0], 95)),
                                                                     (np.percentile(x[:, 1], 5),
                                                                         np.percentile(x[:, 1], 95)),
                                                                     (np.percentile(x[:, 2], 5),
                                                                         np.percentile(x[:, 2], 95))))[0]
        self.map = gaussian_filter(self.map, 5)


class ImageHeterogeneity(HeterogeneityGeneratorBase):
    """
    This heterogeneity generator takes a pre-specified 2D image, currently only supporting numpy arrays, and uses them
    as a map for heterogeneity within the tissue. By default, it will use download and use beef ultrasound images taken
    by our team.

    ##########
    This example will (if not previously downloaded) download a folder with beef ultrasound images
    ##########

    Attributes:
        map: the np array of the heterogeneity map transformed and augments to fill the area
    """

    def __init__(self, xdim: int, ydim: int, zdim: int, spacing_mm: Union[int, float],
                 heterogeneity_image: np.ndarray = None, image_pixel_spacing_mm: Union[int, float] = None,
                 scaling_type: str = Tags.IMAGE_SCALING_SYMMETRIC, constant: Union[int, float] = 0,
                 crop_placement=('centre', 'centre'), target_mean: Union[int, float] = None,
                 target_std: Union[int, float] = None, target_min: Union[int, float] = None,
                 target_max: Union[int, float] = None, beef_ultrasound_database_path: str = None):
        """
        :param xdim: the x dimension of the volume in voxels
        :param ydim: the y dimension of the volume in voxels
        :param zdim: the z dimension of the volume in voxels
        :param heterogeneity_image: the 2D prior image of the heterogeneity map
        :param spacing_mm: the spacing of the volume in mm
        :param image_pixel_spacing_mm: the pixel spacing of the image in mm (pixel spacing)
        :param scaling_type: the scaling type of the heterogeneity map, with default being that no scaling occurs
            OPTIONS:
            Tags.IMAGE_SCALING_SYMMETRIC: symmetric reflections of the image to span the area
            Tags.IMAGE_SCALING_STRETCH: stretch the image to span the area
            Tags.IMAGE_SCALING_WRAP: multiply the image to span the area
            Tags.IMAGE_SCALING_EDGE: continue the values at the edge of the area to fill the shape
            Tags.IMAGE_SCALING_CONSTANT: span the left-over area with a constant
        :param constant: the scaling constant of the heterogeneity map, used only for scaling type 'constant'
            WARNING: scaling constant must be in reference to the values in the heterogeneity_image
        :param crop_placement: the placement of where the heterogeneity map is cropped
        :param target_mean: (optional) the mean of the created heterogeneity map
        :param target_std: (optional) the standard deviation of the created heterogeneity map
        :param target_min: (optional) the minimum of the created heterogeneity map
        :param target_max: (optional) the maximum of the created heterogeneity map
        """
        super().__init__(xdim, ydim, zdim, spacing_mm, target_mean, target_std, target_min, target_max)
        self.logger = Logger()
        self.heterogeneity_image = heterogeneity_image

        if self.heterogeneity_image is None:
            self.heterogeneity_image = self.get_default_ultrasound_image(beef_ultrasound_database_path)
            image_pixel_spacing_mm = 0.2

        if image_pixel_spacing_mm is None:
            image_pixel_spacing_mm = spacing_mm

        if scaling_type == Tags.IMAGE_SCALING_STRETCH:
            self.heterogeneity_image = transform.resize(self.heterogeneity_image, output_shape=(xdim, zdim),
                                                        mode='symmetric')
        else:
            (image_width_pixels, image_height_pixels) = self.heterogeneity_image.shape
            [image_width_mm, image_height_mm] = np.array(
                [image_width_pixels, image_height_pixels]) * image_pixel_spacing_mm
            (xdim_mm, ydim_mm, zdim_mm) = np.array([xdim, ydim, zdim]) * spacing_mm

            wider = image_width_mm > xdim_mm
            taller = image_height_mm > zdim_mm

            if taller or wider:
                self.change_resolution(spacing_mm=spacing_mm, image_pixel_spacing_mm=image_pixel_spacing_mm)
                self.crop_image(xdim, zdim, crop_placement)
                if not taller and not wider:
                    self.upsize_to_fill_area(xdim, zdim, scaling_type, constant)

            else:
                self.change_resolution(spacing_mm=spacing_mm, image_pixel_spacing_mm=image_pixel_spacing_mm)
                self.upsize_to_fill_area(xdim, zdim, scaling_type, constant)

        self.map = np.repeat(self.heterogeneity_image[:, np.newaxis, :], ydim, axis=1)

    def upsize_to_fill_area(self, xdim: int, zdim: int, scaling_type: str = Tags.IMAGE_SCALING_SYMMETRIC,
                            constant: Union[int, float] = 0):
        """
        Fills an area with an image through various methods of expansion
        :param xdim: the x dimension of the area to be filled in voxels
        :param zdim: the z dimension of the area to be filled in voxels
        :param scaling_type: the scaling type of the heterogeneity map, with default being that no scaling occurs
            OPTIONS:
            TAGS.IMAGE_SCALING_SYMMETRIC: symmetric reflections of the image to span the area
            TAGS.IMAGE_SCALING_STRETCH: stretch the image to span the area
            TAGS.IMAGE_SCALING_WRAP: multiply the image to span the area
            TAGS.IMAGE_SCALING_EDGE: continue the values at the edge of the area to fill the shape
            TAGS.IMAGE_SCALING_CONSTANT: span the left-over area with a constant
        :param constant: the scaling constant of the heterogeneity map, used only for scaling type 'constant'
        """
        if scaling_type == Tags.IMAGE_SCALING_STRETCH:
            pass
        elif scaling_type == Tags.IMAGE_SCALING_CONSTANT:
            pad_left = int((xdim - len(self.heterogeneity_image)) / 2)
            pad_height = int(zdim - len(self.heterogeneity_image[0]))
            pad_right = xdim - pad_left - len(self.heterogeneity_image)
            self.heterogeneity_image = np.pad(self.heterogeneity_image, ((pad_left, pad_right), (0, pad_height)),
                                              mode=scaling_type, constant_values=constant)
        else:
            pad_left = int((xdim - len(self.heterogeneity_image)) / 2)
            pad_height = int(zdim - len(self.heterogeneity_image[0]))
            pad_right = xdim - pad_left - len(self.heterogeneity_image)
            self.heterogeneity_image = np.pad(self.heterogeneity_image, ((pad_left, pad_right), (0, pad_height)),
                                              mode=scaling_type)

        self.logger.warning("The input image has filled the area by using {} scaling type".format(scaling_type))

    def crop_image(self, xdim: int, zdim: int,
                   crop_placement: Union[str, tuple] = Tags.CROP_POSITION_CENTRE) -> np.ndarray:
        """
        Crop the image to fit specified dimensions xdim and zdim
        :param xdim: the x dimension of the area to be filled in voxels
        :param zdim: the z dimension of the area to be filled in voxels
        :param crop_placement: the placement of where the heterogeneity map is cropped
            OPTIONS: TAGS.CROP_PLACEMENT_[TOP,BOTTOM,LEFT,RIGHT,CENTRE,RANDOM] or position of left hand corner on image
        """
        (image_width_pixels, image_height_pixels) = self.heterogeneity_image.shape
        crop_width = min(xdim, image_width_pixels)
        crop_height = min(zdim, image_height_pixels)

        if isinstance(crop_placement, tuple):
            if crop_placement[0] == Tags.CROP_POSITION_LEFT:
                crop_horizontal = 0
            elif crop_placement[0] == Tags.CROP_POSITION_RIGHT:
                crop_horizontal = image_width_pixels-crop_width-1
            elif crop_placement[0] == Tags.CROP_POSITION_CENTRE:
                crop_horizontal = round((image_width_pixels - crop_width) / 2)
            elif isinstance(crop_placement[0], int):
                crop_horizontal = crop_placement[0]
            else:
                raise ValueError(f"Invalid crop placement {crop_placement[0]}. Please check Tags.CROP_POSITION_... for"
                                 f"valid string arguments and that numbers are of type int")

            if crop_placement[1] == Tags.CROP_POSITION_TOP:
                crop_vertical = 0
            elif crop_placement[1] == Tags.CROP_POSITION_BOTTOM:
                crop_vertical = image_height_pixels-crop_height-1
            elif crop_placement[1] == Tags.CROP_POSITION_CENTRE:
                crop_vertical = round((image_height_pixels - crop_height) / 2)
            elif isinstance(crop_placement[1], int):
                crop_vertical = crop_placement[1]
            else:
                raise ValueError(f"Invalid crop placement {crop_placement[1]}. Please check Tags.CROP_POSITION_... for"
                                 f"valid string arguments and that numbers are of type int")

        elif isinstance(crop_placement, str):
            if crop_placement == Tags.CROP_POSITION_CENTRE:
                crop_horizontal = round((image_width_pixels - crop_width) / 2)
                crop_vertical = round((image_height_pixels - crop_height) / 2)
            elif crop_placement == Tags.CROP_POSITION_RANDOM:
                crop_horizontal = image_width_pixels - crop_width
                if crop_horizontal != 0:
                    crop_horizontal = np.random.randint(0, crop_horizontal)
                crop_vertical = image_height_pixels - crop_height
                if crop_vertical != 0:
                    crop_vertical = np.random.randint(0, crop_vertical)
            else:
                raise ValueError(f"Invalid crop placement {crop_placement}. Please check Tags.CROP_POSITION_... for"
                                 f"valid arguments")

        else:
            raise ValueError("Crop placement must be tuple or str")

        self.heterogeneity_image = self.heterogeneity_image[crop_horizontal: crop_horizontal +
                                                            crop_width, crop_vertical: crop_vertical + crop_height]

        self.logger.warning(
            "The input image has been cropped to the dimensions of the simulation volume ({} {})".format(xdim, zdim))

    def change_resolution(self, spacing_mm: Union[int, float],
                          image_pixel_spacing_mm: Union[int, float]):
        """
        Method to change the resolution of an image
        :param image_pixel_spacing_mm: original image pixel spacing mm
        :param spacing_mm: target pixel spacing mm
        """
        (image_width_pixels, image_height_pixels) = self.heterogeneity_image.shape
        [image_width_mm, image_height_mm] = np.array([image_width_pixels, image_height_pixels]) * image_pixel_spacing_mm
        new_image_pixel_width = round(image_width_mm / spacing_mm)
        new_image_pixel_height = round(image_height_mm / spacing_mm)

        self.logger.warning(
            "The input image has changed pixel spacing to {} to match the simulation volume".format(spacing_mm))
        self.heterogeneity_image = transform.resize(self.heterogeneity_image, (new_image_pixel_width,
                                                                               new_image_pixel_height))

    @staticmethod
    def get_default_ultrasound_image(beef_ultrasound_database_path):
        logger = Logger()
        if not beef_ultrasound_database_path:
            current_dir = os.getcwd()
            beef_ultrasound_database_path = os.path.join(current_dir, "beef_ultrasound_database")
            if not os.path.exists(beef_ultrasound_database_path):
                download_ultrasound_images(current_dir)
        # make the image be random
        rng = np.random.default_rng()
        scan_number = rng.integers(low=2, high=63)
        logger.debug("Scan number {} was used for this simulation".format(scan_number))
        heterogeneity_image = np.load(beef_ultrasound_database_path + "/Scan_" + str(scan_number) + ".npy")
        return heterogeneity_image


def download_ultrasound_images(save_dir):
    """
    Downloads the latest beef ultrasound images from nextcloud. The metadata about these images can be found in the
    folder

    :return: None
    """
    logger = Logger()
    # nextcloud url with the reference images
    nextcloud_url = "https://hub.dkfz.de/s/g8fLZiY5D6ZC3Hx"  # shared "reference_figures" folder on nextcloud
    # Specify the local directory to save the files
    zip_filepath = os.path.join(save_dir, "downloaded.zip")
    # Construct the download URL based on the public share link
    download_url = nextcloud_url.replace('/s/', '/index.php/s/') + '/download'
    # Send a GET request to download the file
    logger.debug(f'Download folder with reference figures from nextcloud...')
    response = requests.get(download_url)
    if response.status_code == 200:
        # Save the file
        with open(zip_filepath, 'wb') as f:
            f.write(response.content)
        logger.debug(f'File downloaded successfully and stored at {zip_filepath}.')
    else:
        logger.critical(f'Failed to download file. Status code: {response.status_code}')
        raise requests.exceptions.HTTPError(f'Failed to download file. Status code: {response.status_code}')

    # Open the zip file
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        # Extract all the contents into the specified directory
        zip_ref.extractall(save_dir)

    logger.debug(f'Files extracted to {save_dir}')

    # Remove the zip file after extraction
    os.remove(zip_filepath)
    logger.debug(f'{zip_filepath} removed successfully.')
