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

from simpa.utils.path_manager import PathManager
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage import zoom
from simpa.io_handling import load_hdf5
from simpa.core.simulation import simulate
from simpa.utils import Tags, Settings, TISSUE_LIBRARY
from simpa.core import *
from simpa.processing import iterative_qPAI_algorithm as iterative_qpai


class TestqPAIReconstruction:
    """
    This class applies the iterative qPAI reconstruction algorithm on a simple test volume and
    - by visualizing the results - lets the user evaluate if the reconstruction is performed correctly.
    This test reconstruction contains a volume creation and an optical simulation.
    """

    def setUp(self):
        """
        Runs a pipeline consisting of volume creation and optical simulation. The resulting hdf5 file of the
        simple test volume is saved at SAVE_PATH location defined in the path_config.env file.
        """

        path_manager = PathManager()

        self.VOLUME_TRANSDUCER_DIM_IN_MM = 20
        self.VOLUME_PLANAR_DIM_IN_MM = 20
        self.VOLUME_HEIGHT_IN_MM = 20
        self.SPACING = 0.2
        self.RANDOM_SEED = 111
        self.VOLUME_NAME = "TestqPAIReconstructionVolume_" + str(self.RANDOM_SEED)

        np.random.seed(self.RANDOM_SEED)

        # These parameters set the general properties of the simulated volume
        self.general_settings = {
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: self.VOLUME_NAME,
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_PLANAR_DIM_IN_MM,
            Tags.WAVELENGTHS: [700]
        }
        self.settings = Settings(self.general_settings)

        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: True,
            Tags.STRUCTURES: self.create_example_tissue()
        })
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
        })

        # run pipeline including volume creation and optical mcx simulation
        pipeline = [
            ModelBasedVolumeCreator(self.settings),
            McxAdapter(self.settings)
        ]
        simulate(pipeline, self.settings)

    def test_qPAI_reconstruction(self):
        """
        Runs iterative qPAI reconstruction on test volume by accessing the settings dictionaries.
        """

        global_settings = self.settings

        # set component settings of the iterative method
        # if tags are not defined the default is chosen for reconstruction
        component_settings = {
            Tags.NOISE_STD: 0.2,
            Tags.DOWNSCALE_FACTOR: 0.76,
            Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION: False,
            Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER: 8,
            Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL: 1e-5
        }

        file = load_hdf5(global_settings[Tags.SIMPA_OUTPUT_PATH])
        self.wavelength = global_settings[Tags.WAVELENGTH]
        absorption_gt = file["simulations"]["simulation_properties"]["mua"][str(self.wavelength)]

        # if the initial pressure is resampled the ground truth has to be resampled to allow for comparison
        if Tags.DOWNSCALE_FACTOR in component_settings:
            self.absorption_gt = zoom(absorption_gt, component_settings[Tags.DOWNSCALE_FACTOR],
                                      order=1, mode="nearest")
        else:
            self.absorption_gt = zoom(absorption_gt, 0.73, order=1, mode="nearest")  # the default scale is 0.73

        # run the qPAI reconstruction and get reconstructed absorptions at each iteration step
        _, list_absorptions_pred = iterative_qpai.run_iterative_reconstruction(global_settings=global_settings,
                                                                               component_settings=component_settings)
        self.list_reconstructed_absorptions = list_absorptions_pred

    def visualize_test_results(self):
        """
        Performs visualization of reconstruction results to allow for evaluation.
        The resulting figure displays the ground truth absorption coefficients, the corresponding reconstruction
        results and the difference between both for the middle plane in y-z and x-z direction, as well as the
        reconstruction results at each iteration step in x-z direction.
        """

        difference_absorption = self.absorption_gt - self.list_reconstructed_absorptions[-1]

        if np.min(self.absorption_gt) > np.min(self.list_reconstructed_absorptions[-1]):
            cmin = np.min(self.list_reconstructed_absorptions[-1])
        else:
            cmin = np.min(self.absorption_gt)

        if np.max(self.absorption_gt) > np.max(self.list_reconstructed_absorptions[-1]):
            cmax = np.max(self.absorption_gt)
        else:
            cmax = np.max(self.list_reconstructed_absorptions[-1])

        x_pos = int(np.shape(self.absorption_gt)[0] / 2)
        y_pos = int(np.shape(self.absorption_gt)[1] / 2)

        results_x_z = [self.absorption_gt[:, y_pos, :], self.list_reconstructed_absorptions[-1][:, y_pos, :],
                       difference_absorption[:, y_pos, :]]
        results_y_z = [self.absorption_gt[x_pos, :, :], self.list_reconstructed_absorptions[-1][x_pos, :, :],
                       difference_absorption[x_pos, :, :]]

        label = ["Absorption coefficients $\hat\mu_a$", "Reconstruction $\mu_a$", "Difference $\hat\mu_a - \mu_a$"]

        plt.figure(figsize=(20, 15))
        plt.subplots_adjust(hspace=0.5, wspace=0.1)
        plt.suptitle("Iterative qPAI Reconstruction")

        for i, quantity in enumerate(results_y_z):
            plt.subplot(4, math.ceil(len(self.list_reconstructed_absorptions) / 2), i + 1)
            if i == 0:
                plt.ylabel("y-z", fontsize=10)
            plt.imshow(np.rot90(quantity, -1))
            plt.title(label[i], fontsize=10)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.colorbar()
            if i != 2:
                plt.clim(cmin, cmax)
            else:
                plt.clim(np.min(difference_absorption), np.max(difference_absorption))

        for i, quantity in enumerate(results_x_z):
            plt.subplot(4, math.ceil(len(self.list_reconstructed_absorptions) / 2),
                        i + math.ceil(len(self.list_reconstructed_absorptions) / 2) + 1)
            if i == 0:
                plt.ylabel("x-z", fontsize=10)
            plt.imshow(np.rot90(quantity, -1))
            plt.title(label[i], fontsize=10)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.colorbar()
            if i != 2:
                plt.clim(cmin, cmax)
            else:
                plt.clim(np.min(difference_absorption), np.max(difference_absorption))

        for i, quantity in enumerate(self.list_reconstructed_absorptions):
            plt.subplot(4, math.ceil(len(self.list_reconstructed_absorptions) / 2),
                        i + 2*math.ceil(len(self.list_reconstructed_absorptions) / 2) + 1)
            plt.title("Iteration step: " + str(i + 1), fontsize=8)
            plt.imshow(np.rot90(quantity[:, y_pos, :], -1))
            plt.colorbar()
            plt.clim(cmin, cmax)
            plt.axis('off')

        plt.show()
        plt.close()

    def create_example_tissue(self):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and two blood vessels. It is used for volume creation.
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        epidermis_structure = Settings()
        epidermis_structure[Tags.PRIORITY] = 1
        epidermis_structure[Tags.STRUCTURE_START_MM] = [0, 0, 2]
        epidermis_structure[Tags.STRUCTURE_END_MM] = [0, 0, 2.5]
        epidermis_structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
        epidermis_structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        epidermis_structure[Tags.ADHERE_TO_DEFORMATION] = True
        epidermis_structure[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        vessel_structure_1 = Settings()
        vessel_structure_1[Tags.PRIORITY] = 2
        vessel_structure_1[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2.5, 0,
                                                       self.VOLUME_HEIGHT_IN_MM / 2]
        vessel_structure_1[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2.5,
                                                     self.VOLUME_PLANAR_DIM_IN_MM, self.VOLUME_HEIGHT_IN_MM / 2]
        vessel_structure_1[Tags.STRUCTURE_RADIUS_MM] = 1.75
        vessel_structure_1[Tags.STRUCTURE_ECCENTRICITY] = 0.85
        vessel_structure_1[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood_arterial()
        vessel_structure_1[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_structure_1[Tags.ADHERE_TO_DEFORMATION] = True
        vessel_structure_1[Tags.STRUCTURE_TYPE] = Tags.ELLIPTICAL_TUBULAR_STRUCTURE

        vessel_structure_2= Settings()
        vessel_structure_2[Tags.PRIORITY] = 3
        vessel_structure_2[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2, 0,
                                                       self.VOLUME_HEIGHT_IN_MM / 3]
        vessel_structure_2[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                     self.VOLUME_PLANAR_DIM_IN_MM, self.VOLUME_HEIGHT_IN_MM / 3]
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


if __name__ == '__main__':
    test = TestqPAIReconstruction()
    test.setUp()
    test.test_qPAI_reconstruction()
    test.visualize_test_results()













