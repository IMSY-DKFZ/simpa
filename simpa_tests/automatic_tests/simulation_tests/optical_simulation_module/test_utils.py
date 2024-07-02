# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa import Settings, Tags, TISSUE_LIBRARY, define_horizontal_layer_structure_settings, DiskIlluminationGeometry, \
    MCXAdapterReflectance, ModelBasedVolumeCreationAdapter, simulate, PathManager
from simpa.core.simulation_modules.optical_simulation_module.utils import \
    get_spectral_image_from_simulated_reflectances, get_spectral_image_from_optical_simulation


def test_if_get_spectral_image_from_simulated_reflectances_is_called_correct_spectral_image_is_returned():
    # Arrange
    sample_wavelengths = np.arange(400, 600 + 1, 100)
    reflectance_values = {"400": np.array([0.01, 0.34]), "500": np.array([0.04, 0.06, 0.52]), "600": np.array([0.05])}
    reflectance_pos = {"400": np.array([[0, 0, 0],
                                        [2, 1, 0]]),
                       "500": np.array([[0, 1, 0],
                                        [1, 0, 0],
                                        [1, 1, 0]]),
                       "600": np.array([[2, 0, 0]])}
    target_height = 2
    target_width = 3
    expected_spectral_image = np.array([[[0.01, 0.0, 0.0],
                                         [0.0, 0.06, 0.0],
                                         [0.0, 0.0, 0.05]],
                                        [[0.0, 0.04, 0.0],
                                         [0.0, 0.52, 0.0],
                                         [0.34, 0.0, 0.0]]])
    # Act
    actual_spectral_image = get_spectral_image_from_simulated_reflectances(sample_wavelengths,
                                                                           reflectance_values,
                                                                           reflectance_pos,
                                                                           target_height,
                                                                           target_width)
    # Assert
    assert actual_spectral_image.shape == expected_spectral_image.shape
    np.testing.assert_allclose(actual_spectral_image, expected_spectral_image, rtol=1e-8, atol=1e-8)


def test_spectral_image_is_returned_after_optical_simulation():
    dim_volume_x_y_z_mm = [4, 6, 7]
    spacing = 0.5

    target_height = round(dim_volume_x_y_z_mm[1] / spacing)
    target_width = round(dim_volume_x_y_z_mm[0] / spacing)

    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_settings = Settings()
    tissue_settings[Tags.BACKGROUND] = background_dictionary

    tissue_settings["tissue"] = define_horizontal_layer_structure_settings(
        molecular_composition=TISSUE_LIBRARY.muscle(),
        z_start_mm=0,
        thickness_mm=dim_volume_x_y_z_mm[2])

    path_manager = PathManager()
    volume_name = "volume_name"

    general_settings = {
        Tags.RANDOM_SEED: 0,
        Tags.VOLUME_NAME: volume_name,
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: spacing,
        Tags.DIM_VOLUME_Z_MM: dim_volume_x_y_z_mm[2],
        Tags.DIM_VOLUME_X_MM: dim_volume_x_y_z_mm[0],
        Tags.DIM_VOLUME_Y_MM: dim_volume_x_y_z_mm[1],
        Tags.WAVELENGTHS: np.arange(500, 800 + 1, 50),
        Tags.DO_FILE_COMPRESSION: True
    }

    expected_spectral_image_shape = (target_height, target_width, 7)

    settings = Settings(general_settings)

    settings.set_volume_creation_settings({
        Tags.SIMULATE_DEFORMED_LAYERS: True,
        Tags.STRUCTURES: tissue_settings
    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.COMPUTE_DIFFUSE_REFLECTANCE: True,
        Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT: True
    })

    pipeline = [
        ModelBasedVolumeCreationAdapter(settings),
        MCXAdapterReflectance(settings),
    ]

    device = DiskIlluminationGeometry(beam_radius_mm=2,
                                      device_position_mm=np.array(
                                          [dim_volume_x_y_z_mm[0] / 2., dim_volume_x_y_z_mm[1] / 2., 0]))

    simulate(pipeline, settings, device)
    hdf_file_path = path_manager.get_hdf5_file_save_path() + "/" + volume_name + ".hdf5"
    actual_spectral_image = get_spectral_image_from_optical_simulation(hdf_file_path, target_height, target_width)
    assert isinstance(actual_spectral_image, np.ndarray)
    assert actual_spectral_image.shape == expected_spectral_image_shape, actual_spectral_image.shape
    assert actual_spectral_image.max() > 1e-4
