"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import unittest
from unittest.case import expectedFailure
from simpa.utils import Tags, Settings
import simpa as sp
import numpy as np
import os


class TestLinearUnmixing(unittest.TestCase):
    """
    This test is an automatic test, so there is no visual confirmation needed.
    The test result is returned by the logger.
    """

    def setUp(self):
        """
        This function lays the foundation for the generic test.
        """
        self.logger = sp.Logger()
        # TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
        #  point to the correct file in the PathManager().
        self.path_manager = sp.PathManager()
        RANDOM_SEED = 471
        self.WAVELENGTHS = [650, 700, 750, 800, 850, 900]  # the performance is checked using two wavelengths
        # Set general settings which are needed by the linear unmixing component
        general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "LinearUnmixingAutomaticTest_" + str(RANDOM_SEED),
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            # It is sufficient to look at only five voxels
            Tags.SPACING_MM: 1,
            Tags.DIM_VOLUME_Z_MM: 5,
            Tags.DIM_VOLUME_X_MM: 1,
            Tags.DIM_VOLUME_Y_MM: 1,
            Tags.WAVELENGTHS: self.WAVELENGTHS
        }
        self.settings = Settings(general_settings)

        # Define device for simulation
        self.device = sp.PencilBeamIlluminationGeometry()

        # Run empty pipeline simulation to "fill" hdf5 file following usual procedure
        pipeline = []
        sp.simulate(pipeline, self.settings, self.device)

        # # Create arrays containing absorption for both wavelengths
        # # This is hard-coded to provide more control
        # oxy = np.array([2.77, 5.67])
        # deoxy = np.array([7.52, 3.7])
        # # Create absorption data fields (five voxels) for each wavelength
        # test_750_mua = np.array([[[0., oxy[0] + deoxy[0], oxy[0], deoxy[0], 0.7 * oxy[0] + 0.3 * deoxy[0]]]])
        # test_850_mua = np.array([[[0., oxy[1] + deoxy[1], oxy[1], deoxy[1], 0.7 * oxy[1] + 0.3 * deoxy[1]]]])
        #
        # # Load settings from hdf5 file
        # self.file = sp.load_hdf5(self.settings[Tags.SIMPA_OUTPUT_PATH])
        #
        # # Fill settings with absorption data fields
        # mua_dict = {
        #     "750": test_750_mua,
        #     "850": test_850_mua
        # }
        # self.file["simulations"] = {
        #     "simulation_properties": {"mua": mua_dict}
        #     }
        # # Save adapted file in hdf5
        # sp.save_hdf5(self.file, self.settings[Tags.SIMPA_OUTPUT_PATH])

    def test(self):
        """
        This function performs the generic test. The result is returned by the logger.
        """

        self.logger.info("Testing linear unmixing...")

        # Set component settings for linear unmixing
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN]),
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: self.WAVELENGTHS,
            Tags.LINEAR_UNMIXING_NON_NEGATIVE: False
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()

        # Load blood oxygen saturation
        lu_results = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        sO2 = lu_results["sO2"]

        # Perform test by comparing computed with expected sO2
        # The result can differ slightly, but the difference should be smaller than 1e-8
        self.assertTrue(np.allclose(sO2, np.array([[[0, 0.5, 1, 0, 0.7]]]), atol=1e-8), "Linear unmixing test failed")

    def test_non_negative_least_squares(self):
        """
        This function performs the generic test using the non-negative least squares linear unmixing of
        scipy.optimize. The result is returned by the logger.
        """

        self.logger.info("Testing linear unmixing...")

        # Set component settings for linear unmixing
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN]),
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: self.WAVELENGTHS,
            Tags.LINEAR_UNMIXING_NON_NEGATIVE: True
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()

        # Load blood oxygen saturation
        lu_results = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        sO2 = lu_results["sO2"]

        # Perform test by comparing computed with expected sO2
        # The result can differ slightly, but the difference should be smaller than 1e-8
        self.assertTrue(np.allclose(sO2, np.array([[[0, 0.5, 1, 0, 0.7]]]), atol=1e-8), "Linear unmixing with "
                                                                                        "non-negative constraint test failed")

    @expectedFailure
    def test_invalid_wavelengths(self):
        """
        This function tests what happens, if invalid wavelengths were selected for computing the LU.
        Expects that a KeyError is thrown.
        """
        self.logger.info("Testing linear unmixing with invalid wavelengths ...")

        # Set component settings for linear unmixing
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN]),
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: [23, 42] # Test random invalid wavelengths
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()

    def test_subset_of_2_wavelengths(self):
        """
        This function tests what happens, if invalid wavelengths were selected for computing the LU.
        Expects that a KeyError is thrown.
        """
        self.logger.info("Testing linear unmixing with invalid wavelengths ...")

        # Set component settings for linear unmixing
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN]),
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: [800, 900]  # Test random invalid wavelengths
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()

    def test_subset_of_3_wavelengths(self):
        """
        This function tests what happens, if invalid wavelengths were selected for computing the LU.
        Expects that a KeyError is thrown.
        """
        self.logger.info("Testing linear unmixing with invalid wavelengths ...")

        # Set component settings for linear unmixing
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN]),
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: [650, 800, 900]  # Test random invalid wavelengths
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()

    @expectedFailure
    def test_oxygen_saturation_without_hemoglobin(self):
        """
        This function tests what happens, if the oxygen saturation shall be computed but
        neither oxy- nor deoxyhemoglobin were selected.
        Expects that a KeyError is thrown.
        """
        self.logger.info("Testing linear unmixing for oxygen saturation without hemoglobin ...")

        # Set component settings for linear unmixing
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2,
        # however in this test we forget to set the chromophores oxy- and deoxyhemoglobin
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_WATER,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_FAT,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_COPPER_SULPHIDE,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_NICKEL_SULPHIDE,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_MELANIN]),
            Tags.WAVELENGTHS: self.WAVELENGTHS
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()

    def test_with_all_absorbers(self):
        """
        This function tests what happens, if all absorbers are selected.
        Expects that a KeyError is thrown.
        """
        self.logger.info("Testing linear unmixing with invalid wavelengths ...")

        # Set component settings for linear unmixing
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_WATER,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_FAT,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_COPPER_SULPHIDE,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_NICKEL_SULPHIDE,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_MELANIN]),
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: self.WAVELENGTHS
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()

    def tearDown(self):
        # Clean up file after testing
        if (os.path.exists(self.settings[Tags.SIMPA_OUTPUT_PATH]) and
                os.path.isfile(self.settings[Tags.SIMPA_OUTPUT_PATH])):
            # Delete the created file
            os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])
