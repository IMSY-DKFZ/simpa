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
from simpa.utils.calculate import rotation_matrix_between_vectors
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry, PlanarArrayDetectionGeometry, \
    CurvedArrayDetectionGeometry
import os
import inspect
import scipy.io as sio
from scipy.spatial.transform import Rotation
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
        del data_dict[Tags.OPTICAL_MODEL_FLUENCE]
        gc.collect()

        tmp_ac_data = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH], SaveFilePaths.SIMULATION_PROPERTIES)

        PA_device = detection_geometry
        PA_device.check_settings_prerequisites(self.global_settings)
        field_of_view_extent = PA_device.get_field_of_view_extent_mm()
        detector_positions_mm = PA_device.get_detector_element_positions_accounting_for_device_position_mm()

        if not self.component_settings.get(Tags.ACOUSTIC_SIMULATION_3D):
            detectors_are_aligned_along_x_axis = field_of_view_extent[2] == 0 and field_of_view_extent[3] == 0
            detectors_are_aligned_along_y_axis = field_of_view_extent[0] == 0 and field_of_view_extent[1] == 0
            if detectors_are_aligned_along_x_axis or detectors_are_aligned_along_y_axis:
                simulate_2d = True
                axes = (0, 1)
                if detectors_are_aligned_along_y_axis:
                    transducer_plane = int(round((detector_positions_mm[0, 0] / self.global_settings[Tags.SPACING_MM]))) - 1
                    image_slice = np.s_[transducer_plane, :, :]
                else:
                    transducer_plane = int(round((detector_positions_mm[0, 1] / self.global_settings[Tags.SPACING_MM]))) - 1
                    image_slice = np.s_[:, transducer_plane, :]
            else:
                simulate_2d = False
                axes = (0, 2)
                image_slice = np.s_[:]
        else:
            simulate_2d = False
            axes = (0, 2)
            image_slice = np.s_[:]

        wavelength = str(self.global_settings[Tags.WAVELENGTH])
        data_dict[Tags.PROPERTY_SPEED_OF_SOUND] = np.rot90(tmp_ac_data[Tags.PROPERTY_SPEED_OF_SOUND][image_slice], 3, axes=axes)
        data_dict[Tags.PROPERTY_DENSITY] = np.rot90(tmp_ac_data[Tags.PROPERTY_DENSITY][image_slice], 3, axes=axes)
        data_dict[Tags.PROPERTY_ALPHA_COEFF] = np.rot90(tmp_ac_data[Tags.PROPERTY_ALPHA_COEFF][image_slice], 3, axes=axes)
        data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] = np.flip(
            np.rot90(data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE][wavelength][image_slice], axes=axes))

        if simulate_2d:
            detector_positions_mm_2d = np.delete(detector_positions_mm, 1, axis=1)
            detector_positions_mm_2d = np.moveaxis(detector_positions_mm_2d, 1, 0)
            data_dict[Tags.SENSOR_ELEMENT_POSITIONS] = detector_positions_mm_2d[[1, 0]]
            orientations = PA_device.get_detector_element_orientations(self.global_settings)
            angles = np.arccos(np.dot(orientations, np.array([1, 0, 0])))
            data_dict[Tags.PROPERTY_DIRECTIVITY_ANGLE] = angles[::-1]
        else:
            detector_positions_mm = np.moveaxis(detector_positions_mm, 1, 0)
            data_dict[Tags.SENSOR_ELEMENT_POSITIONS] = detector_positions_mm[[2, 1, 0]]
            orientations = PA_device.get_detector_element_orientations(self.global_settings)
            x_angles = np.arccos(np.dot(orientations, np.array([1, 0, 0]))) * 360 / (2*np.pi)
            y_angles = np.arccos(np.dot(orientations, np.array([0, 1, 0]))) * 360 / (2*np.pi)
            z_angles = np.arccos(np.dot(orientations, np.array([0, 0, 1]))) * 360 / (2*np.pi)
            intrinsic_euler_angles = list()
            for orientation_vector in orientations:

                mat = rotation_matrix_between_vectors(orientation_vector, np.array([0, 0, 1]))
                rot = Rotation.from_matrix(mat)
                euler_angles = rot.as_euler("XYZ", degrees=True)
                intrinsic_euler_angles.append(euler_angles)
            intrinsic_euler_angles.reverse()
            angles = np.array([z_angles[::-1], y_angles[::-1], x_angles[::-1]])
            data_dict[Tags.PROPERTY_DIRECTIVITY_ANGLE] = angles
            data_dict[Tags.PROPERTY_INTRINSIC_EULER_ANGLE] = intrinsic_euler_angles

        optical_path = self.global_settings[Tags.SIMPA_OUTPUT_PATH] + ".mat"

        possible_k_wave_parameters = [Tags.SPACING_MM,
                                      Tags.PROPERTY_ALPHA_POWER, Tags.GPU, Tags.PMLInside, Tags.PMLAlpha, Tags.PlotPML,
                                      Tags.RECORDMOVIE, Tags.MOVIENAME, Tags.ACOUSTIC_LOG_SCALE,
                                      Tags.SENSOR_DIRECTIVITY_PATTERN]

        k_wave_settings = Settings({
            Tags.SENSOR_NUM_ELEMENTS: PA_device.number_detector_elements,
            Tags.DETECTOR_ELEMENT_WIDTH_MM: PA_device.detector_element_width_mm,
            Tags.SENSOR_CENTER_FREQUENCY_HZ: PA_device.center_frequency_Hz,
            Tags.SENSOR_BANDWIDTH_PERCENT: PA_device.bandwidth_percent
        })
        if isinstance(PA_device, CurvedArrayDetectionGeometry):
            k_wave_settings[Tags.SENSOR_RADIUS_MM] = PA_device.radius_mm

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

        del data_dict, k_wave_settings, detector_positions_mm, PA_device
        gc.collect()

        if not simulate_2d:
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

        # reverse the order of detector elements from matlab to python order
        raw_time_series_data = raw_time_series_data[::-1, :]

        time_grid = sio.loadmat(optical_path + "dt.mat")
        num_time_steps = int(np.round(time_grid["number_time_steps"]))

        self.global_settings[Tags.K_WAVE_SPECIFIC_DT] = float(time_grid["time_step"])
        self.global_settings[Tags.K_WAVE_SPECIFIC_NT] = num_time_steps

        save_hdf5(self.global_settings, self.global_settings[Tags.SIMPA_OUTPUT_PATH], "/settings/")

        os.remove(optical_path)
        os.remove(optical_path + "dt.mat")
        os.chdir(cur_dir)

        return raw_time_series_data
