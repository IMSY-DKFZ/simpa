# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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

from simpa.utils import Tags, TISSUE_LIBRARY
from simpa.core.simulation import simulate
from simpa.io_handling import load_hdf5
from simpa.utils.settings import Settings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from simpa.simulation_components import VolumeCreationModelModelBasedAdapter, OpticalForwardModelMcxAdapter, GaussianNoiseProcessingComponent
from simpa.processing.iterative_qPAI_algorithm import IterativeqPAIProcessingComponent
from simpa.utils.path_manager import PathManager


# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

VOLUME_TRANSDUCER_DIM_IN_MM = 30
VOLUME_PLANAR_DIM_IN_MM = 30
VOLUME_HEIGHT_IN_MM = 30
SPACING = 0.2
RANDOM_SEED = 471
VOLUME_NAME = "MyqPAIReconstruction_" + str(RANDOM_SEED)

# If VISUALIZE is set to True, the reconstruction result will be plotted
VISUALIZE = True


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    epidermis_structure = Settings()
    epidermis_structure[Tags.PRIORITY] = 1
    epidermis_structure[Tags.STRUCTURE_START_MM] = [0, 0, VOLUME_HEIGHT_IN_MM / 10]
    epidermis_structure[Tags.STRUCTURE_END_MM] = [0, 0, (VOLUME_HEIGHT_IN_MM / 10) + 1]
    epidermis_structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
    epidermis_structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_structure[Tags.ADHERE_TO_DEFORMATION] = True
    epidermis_structure[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    vessel_structure_1 = Settings()
    vessel_structure_1[Tags.PRIORITY] = 2
    vessel_structure_1[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2.5, 0, VOLUME_HEIGHT_IN_MM / 2]
    vessel_structure_1[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2.5, VOLUME_PLANAR_DIM_IN_MM,
                                                 VOLUME_HEIGHT_IN_MM / 2]
    vessel_structure_1[Tags.STRUCTURE_RADIUS_MM] = 2
    vessel_structure_1[Tags.STRUCTURE_ECCENTRICITY] = 0.75
    vessel_structure_1[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood_arterial()
    vessel_structure_1[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_structure_1[Tags.ADHERE_TO_DEFORMATION] = True
    vessel_structure_1[Tags.STRUCTURE_TYPE] = Tags.ELLIPTICAL_TUBULAR_STRUCTURE

    vessel_structure_2 = Settings()
    vessel_structure_2[Tags.PRIORITY] = 3
    vessel_structure_2[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2, 0, VOLUME_HEIGHT_IN_MM / 3]
    vessel_structure_2[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2, VOLUME_PLANAR_DIM_IN_MM,
                                                 VOLUME_HEIGHT_IN_MM / 3]
    vessel_structure_2[Tags.STRUCTURE_RADIUS_MM] = 0.75
    vessel_structure_2[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood_generic()
    vessel_structure_2[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_structure_2[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["epidermis"] = epidermis_structure
    tissue_dict["vessel_1"] = vessel_structure_1
    tissue_dict["vessel_2"] = vessel_structure_2
    return tissue_dict

# set settings for volume creation, optical simulation and iterative qPAI method
np.random.seed(RANDOM_SEED)

general_settings = {
    # These parameters set the general properties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: [700]
}

settings = Settings(general_settings)

settings.set_volume_creation_settings({
    # These parameters set the properties for the volume creation
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.STRUCTURES: create_example_tissue()
})
settings.set_optical_settings({
    # These parameters set the properties for the optical Monte Carlo simulation
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
})
settings["noise_model"] = {
    Tags.NOISE_MEAN: 0.0,
    Tags.NOISE_STD: 0.4,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}
settings["iterative_qPAI_reconstruction"] = {
    # These parameters set the properties of the iterative reconstruction
    Tags.DOWNSCALE_FACTOR: 0.73,
    Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION: False,
    # the following tag has no effect, since the regularization is chosen to be SNR dependent, not constant
    Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA: 0.01,
    Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER: 10,
    # for this example, we are not interested in all absorption updates
    Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS: False,
    Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL: 0.03,
}

# run pipeline including iterative qPAI method
pipeline = [
    VolumeCreationModelModelBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_model"),
    IterativeqPAIProcessingComponent(settings, "iterative_qPAI_reconstruction")
]

simulate(pipeline, settings)

# visualize reconstruction results
if VISUALIZE:
    # get simulation output
    data_path = path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5"
    gt = load_hdf5(data_path)

    # get reconstruction result
    reconstruction_path = path_manager.get_hdf5_file_save_path() + "/Reconstructed_absorption_" + VOLUME_NAME + ".npy"
    absorption_reconstruction = np.load(reconstruction_path)

    # get ground truth absorption coefficients
    settings = Settings(gt["settings"])
    wavelength = settings[Tags.WAVELENGTHS][0]
    absorption_gt = gt["simulations"]["simulation_properties"]["mua"][str(wavelength)]

    # rescale ground truth to same dimension as reconstruction (necessary due to resampling in iterative algorithm)
    scale = np.shape(absorption_reconstruction)[0] / np.shape(absorption_gt)[0]  # same as Tags.DOWNSCALE_FACTOR
    absorption_gt = zoom(absorption_gt, scale, order=1, mode="nearest")

    # compute reconstruction error
    difference = absorption_gt - absorption_reconstruction

    median_error = np.median(difference)
    q3, q1 = np.percentile(difference, [75, 25])
    iqr = q3 - q1

    # visualize results
    x_pos = int(np.shape(absorption_gt)[0] / 2)
    y_pos = int(np.shape(absorption_gt)[1] / 2)

    if np.min(absorption_gt) > np.min(absorption_reconstruction):
        cmin = np.min(absorption_reconstruction)
    else:
        cmin = np.min(absorption_gt)

    if np.max(absorption_gt) > np.max(absorption_reconstruction):
        cmax = np.max(absorption_gt)
    else:
        cmax = np.max(absorption_reconstruction)

    results_x_z = [absorption_gt[:, y_pos, :], absorption_reconstruction[:, y_pos, :], difference[:, y_pos, :]]
    results_y_z = [absorption_gt[x_pos, :, :], absorption_reconstruction[x_pos, :, :], difference[x_pos, :, :]]

    label = ["Absorption coefficients: ${\mu_a}^{gt}$", "Reconstruction: ${\mu_a}^{reconstr.}$",
             "Difference: ${\mu_a}^{gt} - {\mu_a}^{reconstr.}$"]

    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Iterative qPAI Reconstruction \n median error = " + str(np.round(median_error, 4)) +
                 "\n IQR = " + str(np.round(iqr, 4)), fontsize=10)

    for i, quantity in enumerate(results_x_z):
        plt.subplot(2, len(results_x_z), i + 1)
        if i == 0:
            plt.ylabel("x-z", fontsize=10)
        plt.title(label[i], fontsize=10)
        plt.imshow(np.rot90(quantity, -1))
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.colorbar()
        if i != 2:
            plt.clim(cmin, cmax)
        else:
            plt.clim(np.min(difference), np.max(difference))

    for i, quantity in enumerate(results_y_z):
        plt.subplot(2, len(results_x_z), i + len(results_x_z) + 1)
        if i == 0:
            plt.ylabel("y-z", fontsize=10)
        plt.title(label[i], fontsize=10)
        plt.imshow(np.rot90(quantity, -1))
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.colorbar()
        if i != 2:
            plt.clim(cmin, cmax)
        else:
            plt.clim(np.min(difference), np.max(difference))

    plt.show()
    plt.close()
