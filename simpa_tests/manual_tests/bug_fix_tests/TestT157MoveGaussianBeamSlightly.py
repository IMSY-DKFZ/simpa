# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

"""
This Script uses the Lambert-Beer law to test mcx or mcxyz for the correct
attenuation of a photon beam passing through a thin absorbing and/or scattering
slab.

All tests test a scenario, where a pencil source in z-dir in the middle
of the xy-plane emits photons in a 27x27x100 medium.
In the middle of the medium is an absorbing and/or scattering slab of 1 pixel in the xy-plane and some distance in z-dir.
In all tests, the fluence in the middle of the xy-plane and in z-dir the pixels 10 and 90 is measured.
So basically, the fluence after passing the slab with total attenuation (mua+mus) is measured.
For instance, we expect, that the fluence decreases by a factor of e^-1 if mua+mus=0.1 mm^-1 and the slab is 10mm long.

Usage of this script:
The script has to be in the same folder as the mcx executable binary.
If this is met, the script can just be run.

Use the test functions to test the specific cases that are explained in the respective tests.
Please read the description of every test and run them one after the other.
Be aware that by running multiple tests at once, the previous tests are overwritten.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from simpa import MCXAdapter, ModelBasedVolumeCreationAdapter
from simpa.core.device_digital_twins import PhotoacousticDevice, PencilBeamIlluminationGeometry
from simpa.core.simulation import simulate
from simpa.io_handling import load_data_field
from simpa.utils import Tags, Settings, PathManager, TissueLibrary
from simpa_tests.manual_tests import ManualIntegrationTestClass
# FIXME temporary workaround for newest Intel architectures
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestT157MoveGaussianBeamSlightly(ManualIntegrationTestClass):

    def create_example_tissue(self, scattering_value=1e-30, absorption_value=1e-30, anisotropy_value=0.0):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and two blood vessels. It is used for volume creation.
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TissueLibrary().constant(absorption_value,
                                                                                    scattering_value,
                                                                                    anisotropy_value)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        return tissue_dict

    def setup(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        If run on another pc, please adjust the SIMULATION_PATH and MODEL_BINARY_PATH fields.
        :return:
        """

        path_manager = PathManager()
        self.z_dim = 40
        self.xy_dim = 40

        self.settings = Settings({
            Tags.WAVELENGTHS: [800],
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: "AnisotropicScatteringTest",
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: 1,
            Tags.DIM_VOLUME_X_MM: self.xy_dim,
            Tags.DIM_VOLUME_Y_MM: self.xy_dim,
            Tags.DIM_VOLUME_Z_MM: self.z_dim,
            Tags.RANDOM_SEED: 4711
        })

        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
            Tags.MCX_ASSUMED_ANISOTROPY: 0.9
        })

    def teardown(self):
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

    def test_normal_case(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        self.device = PhotoacousticDevice(device_position_mm=np.asarray([self.settings[Tags.DIM_VOLUME_X_MM] / 2,
                                                                         self.settings[Tags.DIM_VOLUME_Y_MM] / 2,
                                                                         0]))

        self.device.add_illumination_geometry(PencilBeamIlluminationGeometry(device_position_mm=np.asarray([0, 0, 0])))

        return self.test_simultion(title="Not moved Illuminator")

    def test_slightly_moved(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        self.device = PhotoacousticDevice(device_position_mm=np.asarray([self.settings[Tags.DIM_VOLUME_X_MM] / 2,
                                                                         self.settings[Tags.DIM_VOLUME_Y_MM] / 2,
                                                                         0]))

        self.device.add_illumination_geometry(PencilBeamIlluminationGeometry(),
                                              illuminator_position_relative_to_pa_device=np.asarray([0, 0, 0.001]))

        return self.test_simultion(title="Moved Illuminator")


    def test_simultion(self, title=""):

        # RUN SIMULATION 1

        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: False,
            Tags.STRUCTURES: self.create_example_tissue(absorption_value=0.1,
                                                        scattering_value=100,
                                                        anisotropy_value=0.9)
        })

        self.settings.get_optical_settings()[Tags.MCX_ASSUMED_ANISOTROPY] = 0.9

        pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings)
        ]

        simulate(pipeline, self.settings, self.device)

        fluence_1 = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_FLUENCE,
                                    self.settings[Tags.WAVELENGTH])

        illuminator_point = int((self.xy_dim / 2) / self.settings[Tags.SPACING_MM]) - 1

        return [title, fluence_1, illuminator_point]


    def visualise_result(self, show_figure_on_screen=True, save_path=None):

        result_1, result_2 = self.results
        (title_1, fluence_1, illuminator_point) = result_1
        (title_2, fluence_2, illuminator_point) = result_2
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.title(title_1)
        plt.imshow(np.log10(fluence_1[:, illuminator_point, :]))
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.title(title_2)
        plt.imshow(np.log10(fluence_2[:, illuminator_point, :]))
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.title("Diff of fluences (lower is better)")
        diff_12 = np.log10(np.abs(fluence_1[:, illuminator_point, :] - fluence_2[:, illuminator_point, :]))
        plt.imshow(diff_12, cmap="Reds", vmax=2, vmin=-4)
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.title("Ratio of fluences (0 is best)")
        diff_13 = np.log10(fluence_1[:, illuminator_point, :] / fluence_2[:, illuminator_point, :])
        plt.imshow(diff_13, vmin=np.min(diff_13), vmax=-np.min(diff_13), cmap="seismic")
        plt.colorbar()

        plt.tight_layout()
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"T157_GaussianBeamPositionTest.png")
        plt.close()

    def perform_test(self):
        self.results = list()
        self.results.append(self.test_normal_case())
        self.results.append(self.test_slightly_moved())


if __name__ == '__main__':
    test = TestT157MoveGaussianBeamSlightly()
    test.run_test(show_figure_on_screen=False)
