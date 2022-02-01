# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os

import numpy as np

import simpa as sp
from simpa import Tags
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
# FIXME temporary workaround for newest Intel architectures
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

# set global params characterizing the simulated volume
VOLUME_TRANSDUCER_DIM_IN_MM = 75
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 25
SPACING = 0.25
RANDOM_SEED = 471
VOLUME_NAME = "LinearUnmixingExample_" + str(RANDOM_SEED)

# since we want to perform linear unmixing, the simulation pipeline should be execute for at least two wavelengths
WAVELENGTHS = [750, 800, 850]


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and two blood vessels.
    """
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    muscle_dictionary = sp.Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle()
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    vessel_1_dictionary = sp.Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 3
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                    10,
                                                    5]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                  12,
                                                  5]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=0.99)
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    vessel_2_dictionary = sp.Settings()
    vessel_2_dictionary[Tags.PRIORITY] = 3
    vessel_2_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/3,
                                                    10,
                                                    5]
    vessel_2_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/3,
                                                  12,
                                                  5]
    vessel_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
    vessel_2_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=0.75)
    vessel_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    epidermis_dictionary = sp.Settings()
    epidermis_dictionary[Tags.PRIORITY] = 8
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 0.1]
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis()
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    tissue_dict["vessel_2"] = vessel_2_dictionary
    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)

# Initialize global settings and prepare for simulation pipeline including
# volume creation and optical forward simulation.
general_settings = {
    # These parameters set the general properties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: WAVELENGTHS,
    Tags.GPU: True,
    Tags.DO_FILE_COMPRESSION: True
}
settings = sp.Settings(general_settings)
settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.STRUCTURES: create_example_tissue()
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
})

# Set component settings for linear unmixing.
# In this example we are only interested in the chromophore concentration of oxy- and deoxyhemoglobin and the
# resulting blood oxygen saturation. We want to perform the algorithm using all three wavelengths defined above.
# Please take a look at the component for more information.
settings["linear_unmixing"] = {
    Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
    Tags.WAVELENGTHS: WAVELENGTHS,
    Tags.LINEAR_UNMIXING_SPECTRA: sp.get_simpa_internal_absorption_spectra_by_names(
        [Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
    ),
    Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
    Tags.LINEAR_UNMIXING_NON_NEGATIVE: True
}

# Get device for simulation
device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     VOLUME_PLANAR_DIM_IN_MM/2,
                                                     0]))
device.update_settings_for_use_of_model_based_volume_creator(settings)

# Run simulation pipeline for all wavelengths in Tag.WAVELENGTHS
pipeline = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
    sp.FieldOfViewCropping(settings),
]
sp.simulate(pipeline, settings, device)

# Run linear unmixing component with above specified settings.
sp.LinearUnmixing(settings, "linear_unmixing").run()

# Load linear unmixing result (blood oxygen saturation) and reference absorption for first wavelength.
file_path = path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5"
lu_results = sp.load_data_field(file_path, Tags.LINEAR_UNMIXING_RESULT)
sO2 = lu_results["sO2"]

mua = sp.load_data_field(file_path, Tags.DATA_FIELD_ABSORPTION_PER_CM, wavelength=WAVELENGTHS[0])
p0 = sp.load_data_field(file_path, Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength=WAVELENGTHS[0])
gt_oxy = sp.load_data_field(file_path, Tags.DATA_FIELD_OXYGENATION, wavelength=WAVELENGTHS[0])

# Visualize linear unmixing result
visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
               wavelength=WAVELENGTHS[0],
               show_initial_pressure=True,
               show_oxygenation=True,
               show_linear_unmixing_sO2=True)