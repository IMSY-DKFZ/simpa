# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
from simpa.utils.profiling import profile
from argparse import ArgumentParser

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that you have set the correct path to MCX binary and SAVE_PATH in the file path_config.env
#  located in the simpa_examples directory


@profile
def run_minimal_optical_simulation(spacing: float | int = 0.5, path_manager=None, visualise: bool = True):
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
    SAVE_REFLECTANCE = False
    SAVE_PHOTON_DIRECTION = False

    # If VISUALIZE is set to True, the simulation result will be plotted

    def create_example_tissue():
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        background_dictionary = sp.Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        muscle_dictionary = sp.Settings()
        muscle_dictionary[Tags.PRIORITY] = 1
        muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 10]
        muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
        muscle_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle()
        muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        vessel_1_dictionary = sp.Settings()
        vessel_1_dictionary[Tags.PRIORITY] = 3
        vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                        10,
                                                        VOLUME_HEIGHT_IN_MM/2]
        vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                      12,
                                                      VOLUME_HEIGHT_IN_MM/2]
        vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
        vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood()
        vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        epidermis_dictionary = sp.Settings()
        epidermis_dictionary[Tags.PRIORITY] = 8
        epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 9]
        epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 10]
        epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis()
        epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        tissue_dict = sp.Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["muscle"] = muscle_dictionary
        tissue_dict["epidermis"] = epidermis_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
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
        Tags.WAVELENGTHS: [798],
        Tags.DO_FILE_COMPRESSION: True,
        Tags.GPU: True
    }

    settings = sp.Settings(general_settings)

    settings.set_volume_creation_settings({
        Tags.SIMULATE_DEFORMED_LAYERS: True,
        Tags.STRUCTURES: create_example_tissue()
    })
    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.COMPUTE_DIFFUSE_REFLECTANCE: SAVE_REFLECTANCE,
        Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT: SAVE_PHOTON_DIRECTION
    })
    settings["noise_model_1"] = {
        Tags.NOISE_MEAN: 1.0,
        Tags.NOISE_STD: 0.1,
        Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
        Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
    }

    if not SAVE_REFLECTANCE and not SAVE_PHOTON_DIRECTION:
        pipeline = [
            sp.ModelBasedAdapter(settings),
            sp.MCXAdapter(settings),
            sp.GaussianNoise(settings, "noise_model_1")
        ]
    else:
        pipeline = [
            sp.ModelBasedAdapter(settings),
            sp.MCXReflectanceAdapter(settings),
        ]

    class ExampleDeviceSlitIlluminationLinearDetector(sp.PhotoacousticDevice):
        """
        This class represents a digital twin of a PA device with a slit as illumination next to a linear detection geometry.

        """

        def __init__(self):
            super().__init__(device_position_mm=np.asarray([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                            VOLUME_PLANAR_DIM_IN_MM/2, 0]))
            self.set_detection_geometry(sp.LinearArrayDetectionGeometry())
            self.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[20, 0, 0],
                                                                       direction_vector_mm=[0, 0, 1]))

    device = ExampleDeviceSlitIlluminationLinearDetector()

    sp.simulate(pipeline, settings, device)

    if Tags.WAVELENGTH in settings:
        WAVELENGTH = settings[Tags.WAVELENGTH]
    else:
        WAVELENGTH = 700

    if visualise:
        sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                          wavelength=WAVELENGTH,
                          show_initial_pressure=True,
                          show_absorption=True,
                          show_diffuse_reflectance=SAVE_REFLECTANCE,
                          log_scale=True)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run the minimal optical simulation example')
    parser.add_argument("--spacing", default=0.2, type=float, help='the voxel spacing in mm')
    parser.add_argument("--path_manager", default=None, help='the path manager, None uses sp.PathManager')
    parser.add_argument("--visualise", default=True, type=bool, help='whether to visualise the result')
    config = parser.parse_args()

    run_minimal_optical_simulation(spacing=config.spacing, path_manager=config.path_manager, visualise=config.visualise)
