# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import numpy as np
from simpa import MCXAdapter,  ModelBasedVolumeCreationAdapter, simulate
from simpa.core.device_digital_twins import PhotoacousticDevice, PencilBeamIlluminationGeometry
from simpa.utils import Settings, Tags, TISSUE_LIBRARY, PathManager
from simpa_tests.manual_tests import ManualIntegrationTestClass


class MCXAdditionalFlags(ManualIntegrationTestClass):

    def create_example_tissue(self):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and two blood vessels. It is used for volume creation.
        """

        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.1, 100, 0.9)
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
        
        self.settings = Settings({
            Tags.WAVELENGTHS: [800],
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: "AdditionalFlagsTest",
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: 1,
            Tags.DIM_VOLUME_X_MM: 100,
            Tags.DIM_VOLUME_Y_MM: 100,
            Tags.DIM_VOLUME_Z_MM: 100,
            Tags.RANDOM_SEED: 4711
        })

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

    def run_simulation(self):
        # run pipeline including volume creation and optical mcx simulation
        pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
        ]
        simulate(pipeline, self.settings, self.device)

    def test_execution_of_additional_flag(self):
        """Tests if log file is created by setting additional parameters

        :raises FileNotFoundError: if log file does not exist at expected location
        """
        output_name = f'{os.path.join(self.settings[Tags.SIMULATION_PATH], self.settings[Tags.VOLUME_NAME])}_output'
        output_file_name = f'{output_name}.log' 

        # perform cleaning before test and checking if file exists afterwards
        if os.path.exists(output_file_name):
            os.remove(output_file_name)
        
        self.settings.get_optical_settings()[Tags.ADDITIONAL_FLAGS] = ['-l', 1, '-s', output_name]
        self.run_simulation()

        if not os.path.exists(output_file_name):
            raise FileNotFoundError(f"Log file wasn't created at expected path {output_file_name}")
        
    def test_if_additional_flag_overrides_existing_flag(self):
        """Tests if log file is created with correct last given name by setting multiple additional parameters

        :raises FileNotFoundError: if correct log file does not exist at expected location
        """
        output_name = f'{os.path.join(self.settings[Tags.SIMULATION_PATH], self.settings[Tags.VOLUME_NAME])}_output'
        output_file_name = f'{output_name}.log' 

        # perform cleaning before test and checking if file exists afterwards
        if os.path.exists(output_file_name):
            os.remove(output_file_name)
        
        self.settings.get_optical_settings()[Tags.ADDITIONAL_FLAGS] = ['-l', 1, '-s', 'temp_name',  '-s', output_name]
        self.run_simulation()

        if not os.path.exists(output_file_name):
            raise FileNotFoundError(f"Log file wasn't created with correct last given name at expected path {output_file_name}")

    def perform_test(self):
        self.test_execution_of_additional_flag()
        self.test_if_additional_flag_overrides_existing_flag()

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        pass
     
if __name__ == '__main__':
    test = MCXAdditionalFlags()
    test.run_test(show_figure_on_screen=False)
