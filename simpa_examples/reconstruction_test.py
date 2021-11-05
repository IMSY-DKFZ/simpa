"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa import Tags
import simpa as sp
import numpy as np

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 75
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 40
SPACING = 0.2
RANDOM_SEED = 4711

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


def create_example_tissue():

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle()
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    tissue_dict["epidermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=1, thickness_mm=0.1,
                                                                             molecular_composition=
                                                                             sp.TISSUE_LIBRARY.epidermis(),
                                                                             priority=8,
                                                                             consider_partial_volume=True,
                                                                             adhere_to_deformation=False)
    tissue_dict["vessel_1"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2 - 10, 0, 9],
        tube_end_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2 - 10, VOLUME_PLANAR_DIM_IN_MM, 9],
        molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=0.9),
        radius_mm=2, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False
    )
    tissue_dict["vessel_2"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/3, 0, 6],
        tube_end_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/3, VOLUME_PLANAR_DIM_IN_MM, 6],
        molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=0.6),
        radius_mm=3, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False
    )
    tissue_dict["vessel_3"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/1.5, 0, 4],
        tube_end_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/1.5, VOLUME_PLANAR_DIM_IN_MM, 4],
        molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=0.6),
        radius_mm=1, priority=4, consider_partial_volume=True,
        adhere_to_deformation=False
    )
    tissue_dict["layer"] = sp.define_rectangular_cuboid_structure_settings(
                                                  start_mm=[40, 6.5, 10], extent_mm=7,
                                                  molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=0.8),
                                                  priority=5,
                                                  consider_partial_volume=True,
                                                  adhere_to_deformation=False
    )

    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

np.random.seed(RANDOM_SEED)
VOLUME_NAME = "CompletePipelineTestMSOT_"+str(RANDOM_SEED)

general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "CompletePipelineTestMSOTtest_" + str(RANDOM_SEED),
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: SPACING,
            Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,
            # The following parameters set the optical forward model
            Tags.WAVELENGTHS: [700],#, 800],
            Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE: True,
            Tags.DO_IPASC_EXPORT: True
        }
settings = sp.Settings(general_settings)
np.random.seed(RANDOM_SEED)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_example_tissue(),
    Tags.SIMULATE_DEFORMED_LAYERS: False
})

settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
})

settings.set_acoustic_settings({
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.PROPERTY_ALPHA_POWER: 0.00,
    Tags.SENSOR_RECORD: "p",
    Tags.PMLInside: False,
    Tags.PMLSize: [31, 32],
    Tags.PMLAlpha: 1.5,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True
})

settings.set_reconstruction_settings({
    Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: True,
    Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
    Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
    Tags.SPACING_MM: settings[Tags.SPACING_MM]
})

settings["noise_initial_pressure"] = {
    Tags.NOISE_MEAN: 1,
    Tags.NOISE_STD: 0.01,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

settings["noise_time_series"] = {
    Tags.NOISE_STD: 1,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.TIME_SERIES_DATA
}

# TODO: For the device choice, uncomment the undesired device

device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     VOLUME_PLANAR_DIM_IN_MM/2,
                                                     0]),
                           field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20]))
device.update_settings_for_use_of_model_based_volume_creator(settings)
#
# device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
#                                                              VOLUME_PLANAR_DIM_IN_MM/2,
#                                                              0]),
#                                 field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20]))
# device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
#                                                               pitch_mm=0.25,
#                                                               number_detector_elements=100,
#                                                               field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20])))
# print(device.get_detection_geometry().get_detector_element_positions_base_mm())
# device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))


SIMUATION_PIPELINE = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
    sp.GaussianNoise(settings, "noise_initial_pressure"),
    sp.KWaveAdapter(settings),
    sp.GaussianNoise(settings, "noise_time_series"),
    sp.DelayAndSumAdapter(settings),
    sp.FieldOfViewCropping(settings)
    ]

sp.simulate(SIMUATION_PIPELINE, settings, device)

if Tags.WAVELENGTH in settings:
    WAVELENGTH = settings[Tags.WAVELENGTH]
else:
    WAVELENGTH = 700

if VISUALIZE:
    sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                      wavelength=WAVELENGTH,
                      show_time_series_data=False,
                      show_initial_pressure=True,
                      show_absorption=True,
                      show_segmentation_map=True,
                      show_tissue_density=False,
                      show_reconstructed_data=True,
                      show_fluence=False,
                      log_scale=True)
