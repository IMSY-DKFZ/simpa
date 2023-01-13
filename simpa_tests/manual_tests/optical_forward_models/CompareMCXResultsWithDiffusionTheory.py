# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags, PathManager, Settings, TISSUE_LIBRARY
from simpa.core.simulation import simulate
from simpa import ModelBasedVolumeCreationAdapter, MCXAdapter
from simpa.core.device_digital_twins import PhotoacousticDevice, PencilBeamIlluminationGeometry
from simpa.io_handling import load_data_field
import numpy as np
import matplotlib.pyplot as plt
from simpa_tests.manual_tests import ManualIntegrationTestClass
# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestCompareMCXResultsWithDiffusionTheory(ManualIntegrationTestClass):

    def create_example_tissue(self):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and two blood vessels. It is used for volume creation.
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(self.mua, self.mus, self.g)
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
        self.dim = 100

        self.settings = Settings({
            Tags.WAVELENGTHS: [800],
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: "DiffuseFluenceTest",
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: 1,
            Tags.DIM_VOLUME_X_MM: self.dim,
            Tags.DIM_VOLUME_Y_MM: self.dim,
            Tags.DIM_VOLUME_Z_MM: self.dim,
            Tags.RANDOM_SEED: 4711
        })

        self.mua = 0.1
        self.mus = 100
        self.g = 0.9

        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: True,
            Tags.STRUCTURES: self.create_example_tissue()
        })
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
            Tags.MCX_ASSUMED_ANISOTROPY: 0.9
        })

        self.device = PhotoacousticDevice(device_position_mm=np.asarray([self.settings[Tags.DIM_VOLUME_X_MM] / 2 - 0.5,
                                                                         self.settings[Tags.DIM_VOLUME_Y_MM] / 2 - 0.5,
                                                                         0]))
        self.device.add_illumination_geometry(PencilBeamIlluminationGeometry())

    def test_fluence(self):
        return self.run_simulation(distance=self.dim / 2, spacing=1)

    def test_spacing_short(self):
        return self.run_simulation(distance=self.dim / 2, spacing=0.333333)

    def test_spacing_middle(self):
        return self.run_simulation(distance=self.dim / 2, spacing=0.5)

    def test_spacing_long(self):
        return self.run_simulation(distance=self.dim / 2, spacing=2)

    def diff_theory_fluence(self, r):
        """
        Calculates the fluence depending on the source-detector distance
        according to the diffusion approximation for the radiative transfer eq.
        right beneath the surface of a semi-finite homogeneous medium.
        :param r: radial distance between source and detector.
        :return: fluence at a point with source-detector distance r.
        """

        mua = 0.1 * self.mua    # convert mua from cm^-1 to mm^-1
        mus = 0.1 * self.mus    # convert mus from cm^-1 to mm^-1
        spacing = self.settings[Tags.SPACING_MM]

        mus_prime = (1-self.g) * mus
        mu_tot = mus_prime + mua
        mu_eff = np.sqrt(3 * mua * mu_tot)
        D = 1 / (3 * (mua + mus_prime))
        z0 = 1 / mu_tot
        n = 1
        r_d = -1.44 * n ** -2 + 0.71 * n ** -1 + 0.668 + 0.0636 * n
        A = (1 + r_d) / (1 - r_d)
        zb = 2 * A * D

        # distance from point source inside the medium to the detector
        detector_distance = np.zeros([len(r), 3])
        detector_distance[:, 0] = r
        detector_distance[:, 2] = 0.5*spacing
        r1 = np.linalg.norm(np.asarray([0, 0, z0]) - detector_distance, axis=1)

        # distance from image point source above the medium to the detector
        r2 = np.linalg.norm(np.asarray([0, 0, -z0 - 2 * zb]) - detector_distance, axis=1)

        # fluence
        phi = 1 / (4*np.pi*D) * (np.exp(-mu_eff*r1) / r1 - np.exp(-mu_eff*r2) / r2)

        return phi

    def run_simulation(self, distance, spacing):

        self.settings[Tags.SPACING_MM] = spacing

        # run pipeline including volume creation and optical mcx simulation
        pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
        ]
        simulate(pipeline, self.settings, self.device)

        return self.assertDiffusionTheory(distance, spacing)

    def perform_test(self):
        self.results = list()
        self.results.append(self.test_spacing_short())
        self.results.append(self.test_spacing_middle())
        self.results.append(self.test_fluence())
        self.results.append(self.test_spacing_long())

    def assertDiffusionTheory(self, distance, spacing):
        fluence = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_FLUENCE,
                                  self.settings[Tags.WAVELENGTH])
        number_of_measurements = np.arange(0, int(distance/self.settings[Tags.SPACING_MM]), 1)
        measurement_distances = number_of_measurements * self.settings[Tags.SPACING_MM]
        fluence_measurements = fluence[int((self.dim/spacing)/2), int((self.dim/spacing)/2) + number_of_measurements, 0]

        fluence_measurements = fluence_measurements / 100

        diffusion_approx = self.diff_theory_fluence(measurement_distances + 1)

        return (measurement_distances, fluence_measurements, diffusion_approx)

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        print(len(self.results))
        for idx, result in enumerate(self.results):
            (measurement_distances, fluence_measurements, diffusion_approx) = result
            fig, ax = plt.subplots()
            ax.scatter(measurement_distances, fluence_measurements, marker="o", c="r", label="Simulation")
            ax.plot(measurement_distances, diffusion_approx, label="Diffusion Approx.")
            ax.fill_between(measurement_distances,
                            diffusion_approx - 0.5 * diffusion_approx,
                            diffusion_approx + 0.5 * diffusion_approx,
                            alpha=0.2, label="Accepted Error Range")
            handles, labels = ax.get_legend_handles_labels()
            handles = [handles[0], handles[2], handles[1]]
            labels = [labels[0], labels[2], labels[1]]
            ax.set_yscale("log")
            plt.legend(handles, labels)

            plt.tight_layout()
            if show_figure_on_screen:
                plt.show()
            else:
                if save_path is None:
                    save_path = ""
                plt.savefig(save_path + f"diffusion_theory_{idx}.png")
            plt.close()


if __name__ == '__main__':
    test = TestCompareMCXResultsWithDiffusionTheory()
    test.run_test(show_figure_on_screen=False)
