# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa.utils.libraries.heterogeneity_generator
from simpa import Tags
import simpa as sp
import numpy as np
from simpa.utils.profiling import profile
from argparse import ArgumentParser

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().


@profile
def run_optical_and_acoustic_simulation(spacing: float | int = 0.2, path_manager=None,
                                        visualise: bool = True):
    """
    ##########
    This example will (if not previously downloaded) download a folder with beef ultrasound images
    ##########

    An example of the full phptoacoustic pipeline and reconstruction with a heterogeneous muscle blood volume fraction
    and the MSOT AcuityEcho.
    The Novelty of this example comes in the origin of its heterogeneous background. Here the heterogeneity come from an
    ultrasound image taken of a piece of beef. For a full description of how the data was obtained, please refeer to the
    md file in the downloaded folder.

    :param spacing: The simulation spacing between voxels
    :param path_manager: the path manager to be used, typically sp.PathManager
    :param visualise: If VISUALIZE is set to True, the reconstruction result will be plotted
    :return: a run through of the example
    """
    if path_manager is None:
        path_manager = sp.PathManager()
    VOLUME_TRANSDUCER_DIM_IN_MM = 60
    VOLUME_PLANAR_DIM_IN_MM = 40
    VOLUME_HEIGHT_IN_MM = 34
    RANDOM_SEED = 4711

    # If VISUALIZE is set to True, the simulation result will be plotted
    VISUALIZE = True

    def create_example_tissue():
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        dim_x, dim_y, dim_z = settings.get_volume_dimensions_voxels()
        background_dictionary = sp.Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        tissue_dict = sp.Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary

        us_heterogeneity = sp.ImageHeterogeneity(xdim=dim_x, ydim=dim_y, zdim=dim_z,
                                                 spacing_mm=spacing, target_min=0, target_max=0.05,
                                                 scaling_type=Tags.IMAGE_SCALING_SYMMETRIC)
        us_heterogeneity.exponential(2)
        us_heterogeneity.invert_image()
        bvf = us_heterogeneity.get_map()

        muscle_dictionary = sp.Settings()
        muscle_dictionary[Tags.PRIORITY] = 1
        muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 34]
        muscle_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle(oxygenation=0.6,
                                                                                blood_volume_fraction=bvf)
        muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        tissue_dict["muscle"] = muscle_dictionary
        return tissue_dict

    # Seed the numpy random configuration prior to creating the global_settings file in
    # order to ensure that the same volume
    # is generated with the same random seed every time.

    np.random.seed(RANDOM_SEED)
    VOLUME_NAME = "CompletePipelineTestMSOT_"+str(RANDOM_SEED)

    general_settings = {
        # These parameters set the general properties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: "CompletePipelineExample_" + str(RANDOM_SEED),
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: spacing,
        Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
        Tags.GPU: True,
        Tags.WAVELENGTHS: [700, 800],
        Tags.DO_FILE_COMPRESSION: True,
        Tags.DO_IPASC_EXPORT: True
    }
    settings = sp.Settings(general_settings)
    np.random.seed(RANDOM_SEED)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_example_tissue(),
        Tags.SIMULATE_DEFORMED_LAYERS: True
    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
        Tags.ADDITIONAL_FLAGS: ['--printgpu']  # to print MCX GPU information
    })

    settings.set_acoustic_settings({
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True
    })

    settings.set_reconstruction_settings({
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(8e6),
        Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(0.1e4),
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
        Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
        Tags.DATA_FIELD_DENSITY: 1000,
        Tags.SPACING_MM: spacing
    })

    settings["noise_initial_pressure"] = {
        Tags.NOISE_MEAN: 1,
        Tags.NOISE_STD: 0.01,
        Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
        Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
    }

    settings["noise_time_series"] = {
        Tags.NOISE_STD: 1,
        Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
    }

    # TODO: For the device choice, uncomment the undesired device

    device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                            VOLUME_PLANAR_DIM_IN_MM/2,
                                                            0]))
    device.update_settings_for_use_of_model_based_volume_creator(settings)

    SIMULATION_PIPELINE = [
        sp.ModelBasedAdapter(settings),
        sp.MCXAdapter(settings),
        sp.GaussianNoise(settings, "noise_initial_pressure"),
        sp.KWaveAdapter(settings),
        sp.GaussianNoise(settings, "noise_time_series"),
        sp.DelayAndSumAdapter(settings),
        sp.FieldOfViewCropping(settings)
    ]

    sp.simulate(SIMULATION_PIPELINE, settings, device)

    if Tags.WAVELENGTH in settings:
        WAVELENGTH = settings[Tags.WAVELENGTH]
    else:
        WAVELENGTH = 700

    if visualise:
        sp.visualise_data(path_to_hdf5_file=settings[Tags.SIMPA_OUTPUT_PATH],
                          wavelength=WAVELENGTH,
                          show_time_series_data=True,
                          show_initial_pressure=True,
                          show_reconstructed_data=True,
                          log_scale=False,
                          show_xz_only=False,
                          show_blood_volume_fraction=True)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run the optical and acoustic simulation example')
    parser.add_argument("--spacing", default=0.2, type=float, help='the voxel spacing in mm')
    parser.add_argument("--path_manager", default=None, help='the path manager, None uses sp.PathManager')
    parser.add_argument("--visualise", default=True, type=bool, help='whether to visualise the result')
    config = parser.parse_args()

    run_optical_and_acoustic_simulation(spacing=config.spacing, path_manager=config.path_manager,
                                        visualise=config.visualise)
