# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from itertools import groupby

from simpa.utils.settings_generator import Settings
from simpa.utils import Tags, SaveFilePaths
from simpa.io_handling import save_hdf5
from simpa.core.volume_creation.versatile_volume_creator import ModelBasedVolumeCreator
from simpa.core.volume_creation.segmentation_based_volume_creator import SegmentationBasedVolumeCreator
from simpa.core.device_digital_twins import DEVICE_MAP
from simpa.utils import create_deformation_settings


def run_volume_creation(global_settings: Settings, wavelength: float):
    """
    This method is the main entry point of volume creation for the SIMPA framework.
    It uses the Tags.VOLUME_CREATOR tag to determine which of the volume creators
    should be used to create the simulation phantom.

    :param global_settings: the settings dictionary that contains the simulation instructions
    :param wavelength: value of wavelength for volume creation
    """
    print("VOLUME CREATION")
    if Tags.CUSTOM_VOLUMES in global_settings and Tags.VOLUME_CREATOR in global_settings:
        raise ValueError(f"Both tags where provided but only one is allowed: "
                         "Tags.CUSTOM_VOLUMES, Tags.VOLUME_CREATOR")
    if Tags.CUSTOM_VOLUMES in global_settings:
        volumes = global_settings[Tags.CUSTOM_VOLUMES][str(wavelength)]
        check_volumes(volumes)
        volume_path = save_volumes_to_file(global_settings, volumes)
        return volume_path

    if Tags.CUSTOM_VOLUMES not in global_settings and Tags.VOLUME_CREATOR not in global_settings:
        raise AssertionError("Tags.VOLUME_CREATOR nor Tags.CUSTOM_VOLUMES tag were not specified in the settings. "
                             "Either one is required. Skipping optical modelling.")

    model = global_settings[Tags.VOLUME_CREATOR]

    if model == Tags.VOLUME_CREATOR_VERSATILE:
        volume_creator_adapter = ModelBasedVolumeCreator()
    elif model == Tags.VOLUME_CREATOR_SEGMENTATION_BASED:
        volume_creator_adapter = SegmentationBasedVolumeCreator()
    else:
        raise ValueError(f"Parsed volume creator unknown: {model}")

    pa_device = None
    if Tags.DIGITAL_DEVICE in global_settings:
        try:
            pa_device = DEVICE_MAP[global_settings[Tags.DIGITAL_DEVICE]]
            pa_device.check_settings_prerequisites(global_settings)
        except KeyError:
            pa_device = None

    if pa_device is not None:
        global_settings = pa_device.adjust_simulation_volume_and_settings(global_settings)

    if Tags.SIMULATE_DEFORMED_LAYERS in global_settings and global_settings[Tags.SIMULATE_DEFORMED_LAYERS]:
        np.random.seed(global_settings[Tags.RANDOM_SEED])
        global_settings[Tags.DEFORMED_LAYERS_SETTINGS] = create_deformation_settings(
            bounds_mm=[[0, global_settings[Tags.DIM_VOLUME_X_MM]],
                       [0, global_settings[Tags.DIM_VOLUME_Y_MM]]],
            maximum_z_elevation_mm=3,
            filter_sigma=0,
            cosine_scaling_factor=1)
        # TODO extract as settings parameters

    volumes = volume_creator_adapter.create_simulation_volume(global_settings)

    volume_path = save_volumes_to_file(global_settings, volumes)

    return volume_path


def save_volumes_to_file(global_settings: dict, volumes: dict) -> str:
    """
    Saves volumes to path defined by:
    `SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.ORIGINAL_DATA, str(global_settings[Tags.WAVELENGTH]))`
    :param global_settings: dictionary containing the settings of the simulations
    :param volumes: dictionary containing the volumes to be saved
    :return: path to saved volumes
    """
    volume_path = SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.ORIGINAL_DATA, str(global_settings[Tags.WAVELENGTH]))
    save_hdf5(volumes, global_settings[Tags.SIMPA_OUTPUT_PATH], file_dictionary_path=volume_path)
    return volume_path


def check_volumes(volumes: dict) -> None:
    """
    Checks consistency of volumes. Helpful when parsing custom volumes to simulations. Checks that all volumes have same
    shape, that the minimum required volumes are defined and that values are meaningful for specific volumes.
    For example: `np.all(volumes['mua'] >= 0)`. It also checks that there are no nan or inf values in all volumes.
    :param volumes: dictionary containing the volumes to check
    :return: None
    """
    shapes = []
    required_vol_keys = Tags.MINIMUM_REQUIRED_VOLUMES
    for key in volumes:
        vol = volumes[key]
        if np.any(np.isnan(vol)) or np.any(np.isinf(vol)):
            raise ValueError(f"Found nan or inf value in volume: {key}")
        if key in ["mua", "mus"] and not np.all(vol >= 0):
            raise ValueError(f"Found negative value in volume: {key}")
        shapes.append(vol.shape)
    g = groupby(shapes)
    shapes_equal = next(g, True) and not next(g, False)
    if not shapes_equal:
        raise ValueError(f"Not all shapes of custom volumes are equal: {shapes}")
    for key in required_vol_keys:
        if key not in volumes.keys():
            raise ValueError(f"Could not find required key in custom volumes {key}")
