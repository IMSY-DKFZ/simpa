"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags, TISSUE_LIBRARY
from simpa.core.simulation import simulate
from simpa.algorithms.multispectral.linear_unmixing import LinearUnmixingProcessingComponent
import numpy as np
from simpa.core import *
from simpa.utils.path_manager import PathManager
from simpa.io_handling import load_data_field
from simpa.core.device_digital_twins import MSOTAcuityEcho
import matplotlib.pyplot as plt
# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

# set global params characterizing the simulated volume
VOLUME_TRANSDUCER_DIM_IN_MM = 75
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 25
SPACING = 0.5
RANDOM_SEED = 471
VOLUME_NAME = "LinearUnmixingExample_" + str(RANDOM_SEED)

# since we want to perform linear unmixing, the simulation pipeline should be execute for at least two wavelengths
WAVELENGTHS = [750, 800, 850]


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    muscle_dictionary = Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    vessel_1_dictionary = Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 3
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                    10,
                                                    5]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                  12,
                                                  5]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood(oxygenation=0.99)
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    epidermis_dictionary = Settings()
    epidermis_dictionary[Tags.PRIORITY] = 8
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 0.1]
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
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

    Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE: True
}
settings = Settings(general_settings)
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
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.WAVELENGTHS: WAVELENGTHS,
    Tags.LINEAR_UNMIXING_OXYHEMOGLOBIN: WAVELENGTHS,
    Tags.LINEAR_UNMIXING_DEOXYHEMOGLOBIN: WAVELENGTHS,
    Tags.LINEAR_UNMIXING_COMPUTE_SO2: True
}

# Get device for simulation
device = MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     VOLUME_PLANAR_DIM_IN_MM/2,
                                                     0]))
device.update_settings_for_use_of_model_based_volume_creator(settings)

# Run simulation pipeline for all wavelengths in Tag.WAVELENGTHS
pipeline = [
    VolumeCreationModelModelBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
    FieldOfViewCroppingProcessingComponent(settings),
]
simulate(pipeline, settings, device)

# Run linear unmixing component with above specified settings.
LinearUnmixingProcessingComponent(settings, "linear_unmixing").run()

# Load linear unmixing result (blood oxygen saturation) and reference absorption for first wavelength.
file_path = path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5"
lu_results = load_data_field(file_path, Tags.LINEAR_UNMIXING_RESULT)
sO2 = lu_results["sO2"]

mua = load_data_field(file_path, Tags.PROPERTY_ABSORPTION_PER_CM, wavelength=WAVELENGTHS[0])
p0 = load_data_field(file_path, Tags.OPTICAL_MODEL_INITIAL_PRESSURE, wavelength=WAVELENGTHS[0])
gt_oxy = load_data_field(file_path, Tags.PROPERTY_OXYGENATION, wavelength=WAVELENGTHS[0])

# Visualize linear unmixing result
data_shape = mua.shape
if len(data_shape) == 3:
    y_dim = int(data_shape[1] / 2)
    p0 = p0[:, y_dim, :]
    mua = mua[:, y_dim, :]
    sO2 = sO2[:, y_dim, :]
    gt_oxy = gt_oxy[:, y_dim, :]

plt.figure(figsize=(15, 5))
plt.suptitle("Linear Unmixing")
plt.subplot(1, 3, 1)
plt.title("Initial Pressure")
plt.imshow(np.rot90(p0, -1))
plt.colorbar(fraction=0.05)
plt.subplot(1, 3, 2)
plt.title("Ground Truth Blood oxygen saturation")
gt_im = plt.imshow(np.rot90(gt_oxy, -1))
plt.colorbar(fraction=0.05)
plt.subplot(1, 3, 3)
plt.title("Blood oxygen saturation")
plt.imshow(np.rot90(sO2, -1))
plt.colorbar(gt_im, fraction=0.05)
plt.show()
