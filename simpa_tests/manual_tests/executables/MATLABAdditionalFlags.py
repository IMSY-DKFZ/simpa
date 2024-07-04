# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import numpy as np
from simpa import MCXAdapter,  ModelBasedVolumeCreationAdapter, simulate, KWaveAdapter
from simpa.core.device_digital_twins import PhotoacousticDevice, PencilBeamIlluminationGeometry, LinearArrayDetectionGeometry
from simpa.utils import Settings, Tags, TISSUE_LIBRARY, PathManager
from simpa_tests.manual_tests import ManualIntegrationTestClass


class MATLABAdditionalFlags(ManualIntegrationTestClass):
    """
    Tests if using Tags.ADDITIONAL_FLAGS to set additional flags for MATLAB works in the KWaveAdapter.
    """

    def create_example_tissue(self):
        """
        Creates a very simple example tissue with only background tissue.
        """

        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.1, 100, 0.9)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary

        return tissue_dict

    def setup(self):
        """
        Creates basic simulation settings and a simulation device.
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
            Tags.RANDOM_SEED: 4711,
            Tags.GPU: True
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
        self.settings.set_acoustic_settings({
            Tags.ACOUSTIC_SIMULATION_3D: False,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True
        })

        self.device = PhotoacousticDevice(device_position_mm=np.asarray([self.settings[Tags.DIM_VOLUME_X_MM] / 2 - 0.5,
                                                                         self.settings[Tags.DIM_VOLUME_Y_MM] / 2 - 0.5,
                                                                         0]))
        self.device.add_illumination_geometry(PencilBeamIlluminationGeometry())
        self.device.set_detection_geometry(LinearArrayDetectionGeometry(device_position_mm=self.device.device_position_mm,
                                                              pitch_mm=0.25,
                                                              number_detector_elements=100,
                                                              field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20])))

        output_name = f'{os.path.join(self.settings[Tags.SIMULATION_PATH], self.settings[Tags.VOLUME_NAME])}'
        self.output_file_name = f'{output_name}.log' 

    def run_simulation(self):
        # run pipeline including volume creation and optical mcx simulation and acoustic matlab kwave simulation
        pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
            KWaveAdapter(self.settings)
        ]
        simulate(pipeline, self.settings, self.device)

    def test_execution_of_additional_flag(self):
        """Tests if log file is created by setting additional parameters

        :raises FileNotFoundError: if log file does not exist at expected location
        """
        
        # perform cleaning before test 
        if os.path.exists(self.output_file_name):
            os.remove(self.output_file_name)
        
        # run simulation
        self.settings.get_acoustic_settings()[Tags.ADDITIONAL_FLAGS] = ['-logfile', self.output_file_name]
        self.run_simulation()

        # checking if file exists afterwards
        if not os.path.exists(self.output_file_name):
            raise FileNotFoundError(f"Log file wasn't created at expected path {self.output_file_name}")
        
    def test_if_last_flag_is_used(self):
        """Tests if log file is created with correct last given name by setting multiple additional parameters

        :raises FileNotFoundError: if correct log file does not exist at expected location
        """
        
        # perform cleaning before test
        if os.path.exists(self.output_file_name):
            os.remove(self.output_file_name)
        
        # run simulation
        self.settings.get_acoustic_settings()[Tags.ADDITIONAL_FLAGS] = ['-logfile', 'temp_name', '-logfile', self.output_file_name]
        self.run_simulation()

        # checking if file exists afterwards
        if not os.path.exists(self.output_file_name):
            raise FileNotFoundError(f"Log file wasn't created with correct last given name at expected path {self.output_file_name}")

    def perform_test(self):
        """
        Calls all individual tests of this class
        """
        self.test_execution_of_additional_flag()
        self.test_if_last_flag_is_used()

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        pass # no figures are created that could be visualized
     
if __name__ == '__main__':
    test = MATLABAdditionalFlags()
    test.run_test(show_figure_on_screen=False)
