# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags, Settings, ModelBasedAdapter
from simpa_tests.test_utils import create_test_structure_parameters
from simpa.core.simulation_modules.optical_module.optical_test_adapter import \
    OpticalTestAdapter
from simpa.core.simulation_modules.acoustic_module.acoustic_test_adapter import \
    AcousticTestAdapter
from simpa.core.simulation import simulate
from simpa.core.device_digital_twins import RSOMExplorerP50

import unittest
import numpy as np
import logging
import os


class TestLaserEnergy(unittest.TestCase):
    """
    Class of tests regarding the laser energy inputted to the model.

    Attributes :
        VOLUME_WIDTH_IN_MM (int | float) : volume dimension (in mm) in the x and y direction
        VOLUME_HEIGHT_IN_MM (int | float) : volume dimension (in mm) in the z direction
        SPACING (int | float) : width of a voxel (in mm)
        RANDOM_SEED (int) : seed of the simulation
    """

    def setUp(self):

        print("setUp")

        self.VOLUME_WIDTH_IN_MM = 4
        self.VOLUME_HEIGHT_IN_MM = 3
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711

    def base_test_case(self, test_name: str, laser_energies: int | np.integer | float | list | range | tuple | np.ndarray):
        """
        Base for all the tests that follow : defines a simple simulation pipeline 
        with optical and acoustic forward simulations.

        :param test_name: name of the performed test
        :type test_name: str
        :param laser_energies: laser energies specified (type varies in the tests)
        :type laser_energies: str

        :returns: launches a simulation
        :rtype: None
        """

        # Configuring the logger :
        # nothing will be displayed except if there is a problem in the simulation
        logger = logging.getLogger("SIMPA Logger")
        logger.setLevel(logging.WARNING)

        np.random.seed(self.RANDOM_SEED)

        settings = {
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: test_name + '_' + str(self.RANDOM_SEED),
            Tags.SIMULATION_PATH: ".",
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_WIDTH_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_WIDTH_IN_MM,
            Tags.WAVELENGTHS: [800, 850, 900],
        }
        settings = Settings(settings)

        settings.set_volume_creation_settings(
            {Tags.STRUCTURES: create_test_structure_parameters()}
        )

        settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_TEST,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: laser_energies
        })

        settings.set_acoustic_settings({})

        pipeline = [
            ModelBasedAdapter(settings),
            OpticalTestAdapter(settings),
            AcousticTestAdapter(settings),
        ]

        try:
            simulate(pipeline, settings, RSOMExplorerP50(0.1, 1, 1))
        except ValueError as e:
            raise e
        finally:
            if (os.path.exists(settings[Tags.SIMPA_OUTPUT_FILE_PATH]) and
                    os.path.isfile(settings[Tags.SIMPA_OUTPUT_FILE_PATH])):
                # Always delete the created file
                os.remove(settings[Tags.SIMPA_OUTPUT_FILE_PATH])

    def test_laser_energy_int(self):
        """
        Checks if we can specify the laser energy as an int, which will be the 
        uniform value for every wavelength.
        Passed if the simulations is launched properly.
        """

        print("Test that the laser energy can be set as an int.")
        self.base_test_case(test_name="TestLaserEnergyInt", laser_energies=10)
        print("PASSED !")

    def test_laser_energy_float(self):
        """
        Checks if we can specify the laser energy as a float, which will be the 
        uniform value for every wavelength.
        Passed if the simulations is launched properly.
        """

        print("Test that the laser energy can be set as a float.")
        self.base_test_case(test_name="TestLaserEnergyFloat", laser_energies=10.5)
        print("PASSED !")

    def test_laser_energy_list(self):
        """
        Checks if we can specify the laser energy as a list of wavelength-dependant 
        energies.
        Passed if the simulations is launched properly.
        """

        print("Test that the laser energy can be set as a list.")
        self.base_test_case(test_name="TestLaserEnergyList", laser_energies=[10, 11, 12])
        print("PASSED !")

    def test_laser_energy_array(self):
        """
        Checks if we can specify the laser energy as an array of wavelength-dependant 
        energies.
        Passed if the simulations is launched properly.
        """

        print("Test that the laser energy can be set as an array.")
        self.base_test_case(test_name="TestLaserEnergyArray", laser_energies=np.array([10, 11, 12]))
        print("PASSED !")

    def test_laser_energy_range(self):
        """
        Checks if we can specify the laser energy as a range of wavelength-dependant 
        energies.
        Passed if the simulations is launched properly.
        """

        print("Test that the laser energy can be set as a range.")
        self.base_test_case(test_name="TestLaserEnergyRange", laser_energies=range(10, 13))
        print("PASSED !")

    def test_laser_energy_tuple(self):
        """
        Checks if we can specify the laser energy as a range of wavelength-dependant 
        energies.
        Passed if the simulations is launched properly.
        """

        print("Test that the laser energy can be set as a range.")
        self.base_test_case(test_name="TestLaserEnergyRange", laser_energies=(10, 11, 12))
        print("PASSED !")

    def test_laser_energy_wrong_size(self):
        """
        Checks if an incorrect size for the laser energy list gives an error.
        Passed if the error is raised correctly.
        """

        print("Test that specifying a laser energy list of wrong size will raise an error.")
        with self.assertRaises(ValueError):
                self.base_test_case(test_name="TestLaserEnergyWrongSize", laser_energies=[10, 11, 12, 13])

        print("PASSED !")