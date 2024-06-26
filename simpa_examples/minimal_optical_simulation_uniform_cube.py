# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT


from simpa import Tags
import simpa as sp
import numpy as np
from typing import Union

# FIXME temporary workaround for newest Intel architectures
import os
from simpa.utils.profiling import profile
from argparse import ArgumentParser

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that you have set the correct path to MCX binary as described in the README.md file.


@profile
def run_minimal_optical_simulation_uniform_cube(spacing: Union[float, int] = 0.5, path_manager=None,
                                                visualise: bool = True):
    """

    :param spacing: The simulation spacing between voxels
    :param path_manager: the path manager to be used, typically sp.PathManager
    :param visualise: If VISUALIZE is set to True, the reconstruction result will be plotted
    :return: a run through of the example
    """
    if path_manager is None:
        path_manager = sp.PathManager()
    VOLUME_TRANSDUCER_DIM_IN_MM = 60
    VOLUME_PLANAR_DIM_IN_MM = 30
    VOLUME_HEIGHT_IN_MM = 60
    RANDOM_SEED = 471
    VOLUME_NAME = "MyVolumeName_"+str(RANDOM_SEED)
    SAVE_REFLECTANCE = True
    SAVE_PHOTON_DIRECTION = False

    # If VISUALIZE is set to True, the simulation result will be plotted
    VISUALIZE = True

    def create_example_tissue():
        """
        This is a very simple example script of how to create a tissue definition.
        It contains only a generic background tissue material.
        """
        background_dictionary = sp.Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        tissue_dict = sp.Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        return tissue_dict

    # Seed the numpy random configuration prior to creating the global_settings file in
    # order to ensure that the same volume
    # is generated with the same random seed every time.

    np.random.seed(RANDOM_SEED)

    general_settings = {
        # These parameters set the general properties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: VOLUME_NAME,
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: spacing,
        Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        Tags.WAVELENGTHS: [500],
        Tags.DO_FILE_COMPRESSION: True
    }

    settings = sp.Settings(general_settings)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_example_tissue()
    })
    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.COMPUTE_DIFFUSE_REFLECTANCE: SAVE_REFLECTANCE,
        Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT: SAVE_PHOTON_DIRECTION
    })

    pipeline = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapterReflectance(settings),
    ]

    device = sp.PencilBeamIlluminationGeometry(device_position_mm=np.asarray([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                              VOLUME_PLANAR_DIM_IN_MM/2, 0]))

    sp.simulate(pipeline, settings, device)

    if visualise:
        sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                          wavelength=settings[Tags.WAVELENGTH],
                          show_initial_pressure=True,
                          show_absorption=True,
                          show_diffuse_reflectance=SAVE_REFLECTANCE,
                          log_scale=True)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run the minimal optical simulation uniform cube example')
    parser.add_argument("--spacing", default=0.2, type=Union[float, int], help='the voxel spacing in mm')
    parser.add_argument("--path_manager", default=None, help='the path manager, None uses sp.PathManager')
    parser.add_argument("--visualise", default=True, type=bool, help='whether to visualise the result')
    config = parser.parse_args()

    spacing = config.spacing
    path_manager = config.path_manager
    visualise = config.visualise
    run_minimal_optical_simulation_uniform_cube(spacing=spacing, path_manager=path_manager, visualise=visualise)
