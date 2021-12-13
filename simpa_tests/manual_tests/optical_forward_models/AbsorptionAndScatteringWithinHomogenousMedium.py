# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
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
from simpa.utils import Tags, Settings, PathManager, TISSUE_LIBRARY
from simpa_tests.manual_tests import ManualIntegrationTestClass
# FIXME temporary workaround for newest Intel architectures
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestAbsorptionAndScatteringWithInifinitesimalSlabExperiment(ManualIntegrationTestClass):

    def create_example_tissue(self, scattering_value=1e-30, absorption_value=1e-30, anisotropy_value=0.0):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and two blood vessels. It is used for volume creation.
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(absorption_value,
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

        self.device = PhotoacousticDevice(device_position_mm=np.asarray([self.settings[Tags.DIM_VOLUME_X_MM] / 2,
                                                                         self.settings[Tags.DIM_VOLUME_Y_MM] / 2,
                                                                         10]))

        self.device.add_illumination_geometry(PencilBeamIlluminationGeometry())

    def teardown(self):
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

    def test_low_scattering(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=1.0,
                                   scattering_value_2=10.0,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.9,
                                   title="Low Abs. Low Scat.")

    def test_medium_scattering(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=10.0,
                                   scattering_value_2=100.0,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.9,
                                   title="Low Abs. Medium Scat.")

    def test_high_scattering_090(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=50.0,
                                   scattering_value_2=500.0,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.9,
                                   title="Anisotropy 0.9")

    def simulate_perfect_result(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=50.0,
                                   scattering_value_2=50.0,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.0,
                                   title="Ideal Result")

    def test_high_scattering_075(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=50.0,
                                   scattering_value_2=200.0,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.75,
                                   title="Anisotropy 0.75")

    def test_high_scattering_025(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=50.0,
                                   scattering_value_2=66.666666666666667,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.25,
                                   title="Anisotropy 0.25")

    def test_ignore_mcx_anisotropy_025(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=50.0,
                                   scattering_value_2=66.666666666666667,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.25,
                                   title="Ignore MCX Anisotropy 0.25",
                                   use_mcx_anisotropy=False)

    def test_ignore_mcx_anisotropy_075(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simultion(absorption_value_1=0.01,
                                   absorption_value_2=0.01,
                                   scattering_value_1=50.0,
                                   scattering_value_2=200.0,
                                   anisotropy_value_1=0.0,
                                   anisotropy_value_2=0.75,
                                   title="Ignore MCX Anisotropy 0.75",
                                   use_mcx_anisotropy=False)

    def test_simultion(self, scattering_value_1=1e-30,
                       absorption_value_1=1e-30,
                       anisotropy_value_1=1.0,
                       scattering_value_2=1e-30,
                       absorption_value_2=1e-30,
                       anisotropy_value_2=1.0,
                       title="Medium Abs. High Scat.",
                       use_mcx_anisotropy=True):

        # RUN SIMULATION 1

        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: False,
            Tags.STRUCTURES: self.create_example_tissue(absorption_value=absorption_value_1,
                                                        scattering_value=scattering_value_1,
                                                        anisotropy_value=anisotropy_value_1)
        })

        self.settings.get_optical_settings()[Tags.MCX_ASSUMED_ANISOTROPY] = anisotropy_value_1

        pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings)
        ]

        simulate(pipeline, self.settings, self.device)

        fluence_1 = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_FLUENCE,
                                    self.settings[Tags.WAVELENGTH])

        # RUN SIMULATION 2

        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: False,
            Tags.STRUCTURES: self.create_example_tissue(absorption_value=absorption_value_2,
                                                        scattering_value=scattering_value_2,
                                                        anisotropy_value=anisotropy_value_2)
        })

        if use_mcx_anisotropy:
            self.settings.get_optical_settings()[Tags.MCX_ASSUMED_ANISOTROPY] = anisotropy_value_2
        else:
            self.settings.get_optical_settings()[Tags.MCX_ASSUMED_ANISOTROPY] = 0.9

        pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings)
        ]

        simulate(pipeline, self.settings, self.device)

        fluence_2 = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_FLUENCE,
                                    self.settings[Tags.WAVELENGTH])

        illuminator_point = int((self.xy_dim / 2) / self.settings[Tags.SPACING_MM]) - 1

        return [title, anisotropy_value_1, scattering_value_1, fluence_1,
                anisotropy_value_2, scattering_value_2, fluence_2, illuminator_point]


    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        print(len(self.results))
        for idx, result in enumerate(self.results):
            (title, anisotropy_value_1, scattering_value_1, fluence_1,
             anisotropy_value_2, scattering_value_2, fluence_2, illuminator_point) = result
            plt.figure(figsize=(10, 8))
            plt.suptitle(title)
            plt.subplot(2, 2, 1)
            plt.title(f"(1) Fluence for (g={anisotropy_value_1}, mus={scattering_value_1})")
            plt.imshow(np.log10(fluence_1[:, illuminator_point, :]))
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.title(f"(2) Fluence for (g={anisotropy_value_2}, mus={scattering_value_2})")
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
                plt.savefig(save_path + f"scattering_test_{idx}.png")
            plt.close()

    def perform_test(self):
        self.results = list()
        self.results.append(self.simulate_perfect_result())
        self.results.append(self.test_low_scattering())
        self.results.append(self.test_medium_scattering())
        self.results.append(self.test_high_scattering_025())
        self.results.append(self.test_high_scattering_075())
        self.results.append(self.test_high_scattering_090())
        self.results.append(self.test_ignore_mcx_anisotropy_025())
        self.results.append(self.test_ignore_mcx_anisotropy_075())


if __name__ == '__main__':
    test = TestAbsorptionAndScatteringWithInifinitesimalSlabExperiment()
    test.run_test(show_figure_on_screen=False)
