"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import numpy as np
import subprocess
from simpa.utils import Tags, SaveFilePaths
from simpa.io_handling.io_hdf5 import load_hdf5, save_hdf5
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.utils.settings import Settings
import os
import inspect
import scipy.io as sio
from simpa.core.acoustic_forward_module import AcousticForwardModelBaseAdapter
import gc


class AcousticForwardModelKWaveAdapter(AcousticForwardModelBaseAdapter):
    """
    The KwaveAcousticForwardModel adapter enables acoustic simulations to be run with the
    k-wave MATLAB toolbox. k-Wave is a free toolbox (http://www.k-wave.org/) developed by Bradley Treeby
    and Ben Cox (University College London) and Jiri Jaros (Brno University of Technology).

    In order to use this toolbox, MATLAB needs to be installed on your system and the path to the
    MATLAB binary needs to be specified in the settings dictionary.

    In order to use the toolbox from with SIMPA, a number of parameters have to be specified in the
    settings dictionary::

        The initial pressure distribution:
            Tags.OPTICAL_MODEL_INITIAL_PRESSURE
        Acoustic tissue properties:
            Tags.PROPERTY_SPEED_OF_SOUND
            Tags.PROPERTY_DENSITY
            Tags.PROPERTY_ALPHA_COEFF
        The digital twin of the imaging device:
            Tags.DIGITAL_DEVICE
        Other parameters:
            Tags.PERFORM_UPSAMPLING
            Tags.SPACING_MM
            Tags.UPSCALE_FACTOR
            Tags.PROPERTY_ALPHA_POWER
            Tags.GPU
            Tags.PMLInside
            Tags.PMLAlpha
            Tags.PlotPML
            Tags.RECORDMOVIE
            Tags.MOVIENAME
            Tags.ACOUSTIC_LOG_SCALE
            Tags.SENSOR_DIRECTIVITY_PATTERN

    Many of these will be set automatically by SIMPA, but you may use the
    simpa.utils.settings_generator convenience methods to generate settings files that contain
    sensible defaults for these parameters.

    Please also refer to the simpa_examples scripts to see how the settings file can be
    parametrized successfully.

    """

    def forward_model(self, detection_geometry) -> np.ndarray:

        optical_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME,
                                          wavelength=self.global_settings[Tags.WAVELENGTH])

        self.logger.debug(f"OPTICAL_PATH: {str(optical_path)}")

        data_dict = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH], optical_path)

        tmp_ac_data = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH], SaveFilePaths.SIMULATION_PROPERTIES)

        wavelength = str(self.global_settings[Tags.WAVELENGTH])
        initial_pressure = data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE][wavelength]

        simulate_3d = (len(np.shape(initial_pressure)) == 3)

        if simulate_3d:
            axes = (0, 2)
        else:
            axes = (0, 1)

        data_dict[Tags.PROPERTY_SPEED_OF_SOUND] = np.rot90(tmp_ac_data[Tags.PROPERTY_SPEED_OF_SOUND], 3, axes=axes)
        data_dict[Tags.PROPERTY_DENSITY] = np.rot90(tmp_ac_data[Tags.PROPERTY_DENSITY], 3, axes=axes)
        data_dict[Tags.PROPERTY_ALPHA_COEFF] = np.rot90(tmp_ac_data[Tags.PROPERTY_ALPHA_COEFF], 3, axes=axes)
        data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] = np.flip(
            np.rot90(initial_pressure, axes=axes))

        try:
            data_dict[Tags.PROPERTY_DIRECTIVITY_ANGLE] = np.rot90(tmp_ac_data[Tags.PROPERTY_DIRECTIVITY_ANGLE], 3,
                                                                  axes=axes)
        except ValueError:
            self.logger.error("No directivity_angle specified")
        except KeyError:
            self.logger.error("No directivity_angle specified")

        del tmp_ac_data
        gc.collect()

        PA_device = detection_geometry
        PA_device.check_settings_prerequisites(self.global_settings)
        detector_positions_mm = PA_device.get_detector_element_positions_accounting_for_device_position_mm(
            self.global_settings)
        # Matlab indexes start at 1
        detector_positions_voxels = np.round(detector_positions_mm / self.global_settings[Tags.SPACING_MM]).astype(int) + 1

        self.logger.debug(f"Number of detector elements: {len(detector_positions_voxels)}")

        if Tags.ACOUSTIC_SIMULATION_3D not in self.component_settings or not self.component_settings[Tags.ACOUSTIC_SIMULATION_3D]:
            sensor_map = detector_positions_voxels[:, [0, 2, 0, 2]]
        else:
            sensor_map = detector_positions_voxels[:, [2, 1, 0, 2, 1, 0]]
        sensor_map = sensor_map.swapaxes(0, 1)
        self.logger.debug(f"Simulated sensor map bounding box coordinates: {str(sensor_map)}")

        self.logger.debug(f"SENSOR_MAP SHAPE: {np.shape(sensor_map)}")

        data_dict[Tags.PROPERTY_SENSOR_MASK] = sensor_map
        save_hdf5({Tags.PROPERTY_SENSOR_MASK: sensor_map}, self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                  generate_dict_path(Tags.PROPERTY_SENSOR_MASK,
                                     wavelength=self.global_settings[Tags.WAVELENGTH]))

        optical_path = self.global_settings[Tags.SIMPA_OUTPUT_PATH] + ".mat"

        possible_k_wave_parameters = [Tags.SPACING_MM,
                                      Tags.PROPERTY_ALPHA_POWER, Tags.GPU, Tags.PMLInside, Tags.PMLAlpha, Tags.PlotPML,
                                      Tags.RECORDMOVIE, Tags.MOVIENAME, Tags.ACOUSTIC_LOG_SCALE,
                                      Tags.SENSOR_DIRECTIVITY_PATTERN]

        k_wave_settings = Settings({
            Tags.SENSOR_NUM_ELEMENTS: PA_device.number_detector_elements,
            Tags.SENSOR_DIRECTIVITY_SIZE_M: PA_device.detector_element_width_mm/1000,
            Tags.SENSOR_CENTER_FREQUENCY_HZ: PA_device.center_frequency_Hz,
            Tags.SENSOR_BANDWIDTH_PERCENT: PA_device.bandwidth_percent
        })

        for parameter in possible_k_wave_parameters:
            if parameter in self.component_settings:
                k_wave_settings[parameter] = self.component_settings[parameter]
                self.logger.debug(f"Added parameter {parameter} to kWave settings via component_settings")
            elif parameter in self.global_settings:
                k_wave_settings[parameter] = self.global_settings[parameter]
                self.logger.debug(f"Added parameter {parameter} to kWave settings via global_settings")
            else:
                self.logger.warning(f"Did not find parameter {parameter} in any settings.")

        data_dict["settings"] = k_wave_settings
        sio.savemat(optical_path, data_dict, long_field_names=True)

        del data_dict, k_wave_settings, sensor_map, detector_positions_voxels, detector_positions_mm, PA_device
        gc.collect()

        if simulate_3d:
            self.logger.info("Simulating 3D....")
            simulation_script_path = "simulate_3D"
        else:
            self.logger.info("Simulating 2D....")
            simulation_script_path = "simulate_2D"

        base_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        cmd = list()
        cmd.append(self.component_settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
        cmd.append("-nodisplay")
        cmd.append("-nosplash")
        cmd.append("-automation")
        cmd.append("-wait")
        cmd.append("-r")
        cmd.append("addpath('" + base_script_path + "');" +
                   simulation_script_path + "('" + optical_path + "');exit;")
        cur_dir = os.getcwd()
        os.chdir(self.global_settings[Tags.SIMULATION_PATH])
        self.logger.info(cmd)
        subprocess.run(cmd)

        raw_time_series_data = sio.loadmat(optical_path)[Tags.TIME_SERIES_DATA]

        time_grid = sio.loadmat(optical_path + "dt.mat")
        num_time_steps = int(np.round(time_grid["number_time_steps"]))

        # TODO create a flag in the PA device specification if output should be 2D or
        #  3D also returns the axis of the imaging plane
        # if (Tags.ACOUSTIC_SIMULATION_3D in self.component_settings and
        #         self.component_settings[Tags.ACOUSTIC_SIMULATION_3D] and
        #     Tags.DIGITAL_DEVICE in self.global_settings and self.global_settings[Tags.DIGITAL_DEVICE] == Tags.DIGITAL_DEVICE_MSOT):
        #
        #     sensor_mask = data_dict[Tags.PROPERTY_SENSOR_MASK]
        #     num_imaging_plane_sensors = int(np.sum(sensor_mask[:, detector_positions_voxels[0][1], :]))
        #
        #     raw_time_series_data = np.reshape(raw_time_series_data, [num_imaging_plane_sensors, -1, num_time_steps])
        #     raw_time_series_data = np.squeeze(np.average(raw_time_series_data, axis=1))

        self.global_settings[Tags.K_WAVE_SPECIFIC_DT] = float(time_grid["time_step"])
        self.global_settings[Tags.K_WAVE_SPECIFIC_NT] = num_time_steps

        save_hdf5(self.global_settings, self.global_settings[Tags.SIMPA_OUTPUT_PATH], "/settings/")

        os.remove(optical_path)
        os.remove(optical_path + "dt.mat")
        os.chdir(cur_dir)

        return raw_time_series_data
