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

    def create_example_tissue(self, slab_width, scattering_value=1e-30, absorption_value=1e-30, anisotropy_value=0.0):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and two blood vessels. It is used for volume creation.
        """
        spacing = self.settings[Tags.SPACING_MM]

        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(self.mua, self.mus, self.g)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        slab_dictionary = Settings()
        slab_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(absorption_value, scattering_value,
                                                                             anisotropy_value)
        slab_dictionary[Tags.STRUCTURE_TYPE] = Tags.RECTANGULAR_CUBOID_STRUCTURE
        slab_dictionary[Tags.PRIORITY] = 9
        slab_dictionary[Tags.STRUCTURE_START_MM] = [(self.xy_dim / 2) - spacing,
                                                    (self.xy_dim / 2) - spacing,
                                                    self.z_dim/2-slab_width/2]
        slab_dictionary[Tags.STRUCTURE_X_EXTENT_MM] = spacing
        slab_dictionary[Tags.STRUCTURE_Y_EXTENT_MM] = spacing
        slab_dictionary[Tags.STRUCTURE_Z_EXTENT_MM] = slab_width
        slab_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        slab_dictionary[Tags.ADHERE_TO_DEFORMATION] = False

        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["slab"] = slab_dictionary

        return tissue_dict

    def setup(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        If run on another pc, please adjust the SIMULATION_PATH and MODEL_BINARY_PATH fields.
        :return:
        """

        path_manager = PathManager()
        self.z_dim = 40
        self.xy_dim = 1

        self.settings = Settings({
            Tags.WAVELENGTHS: [800],
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: "DiffuseFluenceTest",
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: 0.1,
            Tags.DIM_VOLUME_X_MM: self.xy_dim,
            Tags.DIM_VOLUME_Y_MM: self.xy_dim,
            Tags.DIM_VOLUME_Z_MM: self.z_dim,
            Tags.RANDOM_SEED: 4711
        })

        self.mua = 1e-30
        self.mus = 1e-30
        self.g = 1.0

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
                                                                         0]))

        self.device.add_illumination_geometry(PencilBeamIlluminationGeometry())

    def teardown(self):
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

    def test_both(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simulation(distance=10, expected_decay_ratio=np.e ** 1, scattering_value=0.5, absorption_value=0.5,
                             anisotropy_value=0.0, title="Absorption and Scattering over 1 cm")

    def test_both_double_width(self):
        """
        Here, the slab is 20 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        We expect a decay ratio of e^2.
        """
        return self.test_simulation(distance=20, expected_decay_ratio=np.e ** 2, scattering_value=0.5, absorption_value=0.5,
                             anisotropy_value=0.0, title="Absorption and Scattering over 2 cm")

    def test_isotropic_scattering(self):
        """
        Here, the slab is 10 mm long, only mus is used with a value of 0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simulation(distance=10, expected_decay_ratio=np.e, scattering_value=1, anisotropy_value=0.0,
                             title="Isotropic Scattering over 1 cm")

    def test_isotropic_scattering_double_width(self):
        """
        Here, the slab is 20 mm long, only mus is used with a value of 0.1 mm^-1.
        We expect a decay ratio of e^2.
        """
        return self.test_simulation(distance=20, expected_decay_ratio=np.e ** 2, scattering_value=1, anisotropy_value=0.0,
                             title="Isotropic Scattering over 2 cm")

    def test_anisotropic_scattering_0_9(self):
        """
        Here, the slab is 10 mm long, only mus is used with a value of 0.1 mm^-1.
        The anisotropy of the scattering is 0.9.
        We expect a decay ratio of e^1.
        """
        return self.test_simulation(distance=10, expected_decay_ratio=np.e, scattering_value=1, anisotropy_value=0.9,
                             title="Anisotropic Scattering (0.9) over 1 cm")

    def test_anisotropic_scattering_0_5(self):
        """
        Here, the slab is 10 mm long, only mus is used with a value of 0.1 mm^-1.
        The anisotropy of the scattering is 0.9.
        We expect a decay ratio of e^1.
        """
        return self.test_simulation(distance=10, expected_decay_ratio=np.e, scattering_value=1, anisotropy_value=0.5,
                             title="Anisotropic Scattering (0.5) over 1 cm")

    def test_anisotropic_scattering_0_1(self):
        """
        Here, the slab is 10 mm long, only mus is used with a value of 0.1 mm^-1.
        The anisotropy of the scattering is 0.9.
        We expect a decay ratio of e^1.
        """
        return self.test_simulation(distance=10, expected_decay_ratio=np.e, scattering_value=1, anisotropy_value=0.1,
                             title="Anisotropic Scattering (0.1) over 1 cm")

    def test_absorption(self):
        """
        Here, the slab is 10 mm long, only mua is used with a value of 0.1 mm^-1.
        We expect a decay ratio of e^1.
        """
        return self.test_simulation(distance=10, expected_decay_ratio=np.e, absorption_value=1,
                             title="Absorption over 1 cm"
                             )

    def test_absorption_double_width(self):
        """
        Here, the slab is 20 mm long, only mua is used with a value of 0.1 mm^-1.
        We expect a decay ratio of e^2.
        """
        return self.test_simulation(distance=20, expected_decay_ratio=np.e ** 2, absorption_value=1,
                             title="Absorption over 2 cm")

    def test_simulation(self, distance=10, expected_decay_ratio=np.e, scattering_value=1e-30,
                        absorption_value=1e-30, anisotropy_value=1.0, title=""):

        # Define the volume of the thin slab

        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: False,
            Tags.STRUCTURES: self.create_example_tissue(distance, absorption_value=absorption_value,
                                                        scattering_value=scattering_value,
                                                        anisotropy_value=anisotropy_value)
        })

        # TODO: Check this when anisotropy values is 1.0, NaN values appear in scattering in such case
        # self.settings.get_optical_settings()[Tags.MCX_ASSUMED_ANISOTROPY] = anisotropy_value

        pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings)
        ]

        simulate(pipeline, self.settings, self.device)

        # run_optical_forward_model(self.settings)
        fluence = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_FLUENCE,
                                  self.settings[Tags.WAVELENGTH])
        absorption = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_ABSORPTION_PER_CM,
                                     self.settings[Tags.WAVELENGTH])
        scattering = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_SCATTERING_PER_CM,
                                     self.settings[Tags.WAVELENGTH])
        anisotropy = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_ANISOTROPY,
                                     self.settings[Tags.WAVELENGTH])

        early_point = int((self.z_dim / 2 - distance / 2) / self.settings[Tags.SPACING_MM])
        late_point = int((self.z_dim / 2 + distance / 2) / self.settings[Tags.SPACING_MM])
        illuminator_point = int((self.xy_dim / 2) / self.settings[Tags.SPACING_MM]) - 1

        print("early fluence", fluence[illuminator_point, illuminator_point, early_point])
        print("late fluence", fluence[illuminator_point, illuminator_point, late_point])
        decay_ratio = fluence[illuminator_point, illuminator_point, early_point] / \
                      fluence[illuminator_point, illuminator_point, late_point]

        expected_end_fluence = fluence[illuminator_point, illuminator_point, early_point] / expected_decay_ratio
        print("Expected", expected_decay_ratio, "and was", decay_ratio)

        return (title, fluence, illuminator_point, expected_end_fluence, absorption,
             scattering, anisotropy)

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        print(len(self.results))
        for idx, result in enumerate(self.results):
            (title, fluence, illuminator_point, expected_end_fluence, absorption,
             scattering, anisotropy) = result

            plt.figure(figsize=(6, 4))
            plt.title(f"Fluence profile for {title}")
            plt.plot(fluence[illuminator_point, illuminator_point, :], label="Fluence")
            plt.axhline(expected_end_fluence, label="Expected Value after Slab", color="red")
            plt.legend(loc="center left")
            ax2 = plt.twinx()
            ax2.plot(absorption[illuminator_point, illuminator_point, :], label="Absorption", linestyle="dashed", alpha=0.5)
            ax2.plot(scattering[illuminator_point, illuminator_point, :], label="Scattering", linestyle="dashed", alpha=0.5)
            ax2.plot(anisotropy[illuminator_point, illuminator_point, :], label="Anisotropy", linestyle="dashed", alpha=0.5)
            plt.legend(loc="center right")

            plt.tight_layout()
            if show_figure_on_screen:
                plt.show()
            else:
                if save_path is None:
                    save_path = ""
                plt.savefig(save_path + f"infinitessimal_slab_{idx}.png")
            plt.close()

    def perform_test(self):
        self.results = list()
        self.results.append(self.test_both())
        self.results.append(self.test_both_double_width())
        self.results.append(self.test_absorption())
        self.results.append(self.test_absorption_double_width())
        self.results.append(self.test_isotropic_scattering())
        self.results.append(self.test_anisotropic_scattering_0_9())
        self.results.append(self.test_anisotropic_scattering_0_5())
        self.results.append(self.test_anisotropic_scattering_0_1())
        self.results.append(self.test_isotropic_scattering_double_width())


if __name__ == '__main__':
    test = TestAbsorptionAndScatteringWithInifinitesimalSlabExperiment()
    test.run_test(show_figure_on_screen=False)
