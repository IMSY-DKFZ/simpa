# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from unittest.case import expectedFailure
from simpa.utils import Tags, Settings
from simpa_tests.test_utils.tissue_models import create_simple_tissue_model
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
            Tags.DIM_VOLUME_Z_MM: 50,
            Tags.DIM_VOLUME_X_MM: 50,
            Tags.DIM_VOLUME_Y_MM: 20,
            Tags.WAVELENGTHS: self.WAVELENGTHS
        }
        self.settings = Settings(general_settings)

        self.settings.set_volume_creation_settings({
            Tags.STRUCTURES: create_simple_tissue_model(transducer_dim_in_mm=50, planar_dim_in_mm=50),
            Tags.SIMULATE_DEFORMED_LAYERS: False
        })

        # Define device for simulation
        self.device = sp.PencilBeamIlluminationGeometry()

        # Run empty pipeline simulation to "fill" hdf5 file following usual procedure
        pipeline = [sp.ModelBasedVolumeCreationAdapter(self.settings)]
        sp.simulate(pipeline, self.settings, self.device)

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
        self.assert_correct_so2_vales(lu_results["sO2"])

    def assert_correct_so2_vales(self, estimates, tolerance=1e-8):
        ground_truth = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_OXYGENATION)
        msg = f"expected {estimates.reshape((-1,))[0]} but was {ground_truth.reshape((-1,))[0]}"
        self.logger.info(msg)
        self.assertTrue(np.allclose(estimates, ground_truth, atol=tolerance), msg=msg)

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
        self.assert_correct_so2_vales(lu_results["sO2"])

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

        # assert correct estimates
        lu_results = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        self.assert_correct_so2_vales(lu_results["sO2"])

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
            Tags.WAVELENGTHS: [650, 800, 850]  # Test random invalid wavelengths
        }

        # Run linear unmixing component
        lu = sp.LinearUnmixing(self.settings, "linear_unmixing")
        lu.run()
        lu_results = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        self.assert_correct_so2_vales(lu_results["sO2"])

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
        lu_results = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        self.assert_correct_so2_vales(lu_results["sO2"], tolerance=1e-2)

    def tearDown(self):
        # Clean up file after testing
        if (os.path.exists(self.settings[Tags.SIMPA_OUTPUT_PATH]) and
                os.path.isfile(self.settings[Tags.SIMPA_OUTPUT_PATH])):
            # Delete the created file
            os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])
