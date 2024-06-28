# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import pathlib
import typing

import numpy as np

from simpa import load_data_field, Tags


def get_spectral_image_from_optical_simulation(simulation_hdf_file_path: typing.Union[str, pathlib.Path]) -> np.ndarray:
    """
    Returns the spectral image from the simulated reflectance data.
    :param simulation_hdf_file_path: The path to the file containing the MCX simulation results.

    :return: The spectral image in H x W x C.
    """
    refl_by_wavelength = load_data_field(simulation_hdf_file_path, Tags.DATA_FIELD_DIFFUSE_REFLECTANCE)
    refl_pos_by_wavelength = load_data_field(simulation_hdf_file_path, Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS)

    assert refl_by_wavelength.keys() == refl_pos_by_wavelength.keys(), \
        (f"The reflectance values contain different wavelengths\n({refl_by_wavelength.keys()})\n"
         f"than the reflectance positions\n({refl_pos_by_wavelength.keys()})\n!")
    wavelengths_in_nm = [int(w) for w in refl_by_wavelength.keys()]

    spacing_mm = load_data_field(simulation_hdf_file_path, Tags.SPACING_MM)
    dim_volume_x_mm = load_data_field(simulation_hdf_file_path, Tags.DIM_VOLUME_X_MM)
    dim_volume_y_mm = load_data_field(simulation_hdf_file_path, Tags.DIM_VOLUME_Y_MM)
    target_height = round(dim_volume_y_mm / spacing_mm)
    target_width = round(dim_volume_x_mm / spacing_mm)

    return get_spectral_image_from_simulated_reflectances(wavelengths_in_nm=wavelengths_in_nm,
                                                          refl_by_wavelength=refl_by_wavelength,
                                                          refl_pos_by_wavelength=refl_pos_by_wavelength,
                                                          target_height=target_height,
                                                          target_width=target_width)


def get_spectral_image_from_simulated_reflectances(wavelengths_in_nm: typing.Union[np.ndarray, list[int]],
                                                   refl_by_wavelength: dict[str, np.ndarray],
                                                   refl_pos_by_wavelength: dict[str, np.ndarray],
                                                   target_height: int, target_width: int) -> np.ndarray:
    """
    Returns the spectral image from the simulated reflectance data.
    :param wavelengths_in_nm: The wavelengths in mm.
    :param refl_by_wavelength: The reflectance values by wavelength from the simulation.
    :param refl_pos_by_wavelength: The reflectance positions in [x, y, z] by wavelength from the simulation.
    :param target_height: The height of the spectral image in voxels.
    :param target_width: The width of the spectral image in voxels.

    :return: The spectral image in H x W x C.
    """
    spectral_image = np.zeros((len(wavelengths_in_nm), target_height, target_width))

    for j, wavelength in enumerate(wavelengths_in_nm):
        reflectances = refl_by_wavelength[str(wavelength)]
        reflectance_positions = refl_pos_by_wavelength[str(wavelength)]
        assert reflectance_positions.shape[0] == reflectances.shape[0] and reflectance_positions.shape[
            1] == 3, reflectance_positions.shape
        reflectance_positions[:, 2] = j  # Width x Height x Channel
        reflectance_positions = reflectance_positions[:, [1, 0, 2]]
        spectral_image[reflectance_positions[:, 0], reflectance_positions[:, 1],
                       reflectance_positions[:, 2]] = reflectances

    return spectral_image
