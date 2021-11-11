# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils.path_manager import PathManager
import numpy as np
import subprocess
from simpa.utils import Tags
from simpa.io_handling.io_hdf5 import load_hdf5, save_hdf5, load_data_field
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.utils.settings import Settings
from simpa.utils.calculate import rotation_matrix_between_vectors
from simpa.core.device_digital_twins import CurvedArrayDetectionGeometry, DetectionGeometryBase
import os
import inspect
import scipy.io as sio
from scipy.spatial.transform import Rotation
from simpa.core.simulation_modules.acoustic_forward_module import AcousticForwardModelBaseAdapter
import gc


class KWaveAdapter(AcousticForwardModelBaseAdapter):
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

    def forward_model(self, detection_geometry: DetectionGeometryBase) -> np.ndarray:
        """
        Runs the acoustic forward model and performs reading parameters and values from an hdf5 file
        before calling the actual algorithm and saves the updated settings afterwards.

        :param detection_geometry:
        :return: simulated time series data (numpy array)

        """

        optical_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME,
                                          wavelength=self.global_settings[Tags.WAVELENGTH])

        self.logger.debug(f"OPTICAL_PATH: {str(optical_path)}")

        data_dict = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH], optical_path)
        if Tags.DATA_FIELD_FLUENCE in data_dict:
            del data_dict[Tags.DATA_FIELD_FLUENCE]
        gc.collect()

        tmp_ac_data = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.SIMULATION_PROPERTIES,
                                      self.global_settings[Tags.WAVELENGTH])

        pa_device = detection_geometry
        pa_device.check_settings_prerequisites(self.global_settings)
        field_of_view_extent = pa_device.field_of_view_extent_mm
        detector_positions_mm = pa_device.get_detector_element_positions_accounting_for_device_position_mm()
        self.logger.debug(f"field_of_view_extent: {field_of_view_extent}")
        
        detectors_are_aligned_along_x_axis = field_of_view_extent[2] == 0 and field_of_view_extent[3] == 0
        detectors_are_aligned_along_y_axis = field_of_view_extent[0] == 0 and field_of_view_extent[1] == 0
        if detectors_are_aligned_along_x_axis or detectors_are_aligned_along_y_axis:
            axes = (0, 1)
            if detectors_are_aligned_along_y_axis:
                transducer_plane = int(round((detector_positions_mm[0, 0] / self.global_settings[Tags.SPACING_MM]))) - 1
                image_slice = np.s_[transducer_plane, :, :]
            else:
                transducer_plane = int(round((detector_positions_mm[0, 1] / self.global_settings[Tags.SPACING_MM]))) - 1
                image_slice = np.s_[:, transducer_plane, :]
        else:
            axes = (0, 2)
            image_slice = np.s_[:]
        
        wavelength = str(self.global_settings[Tags.WAVELENGTH])
        data_dict[Tags.DATA_FIELD_SPEED_OF_SOUND] = np.rot90(tmp_ac_data[Tags.DATA_FIELD_SPEED_OF_SOUND][image_slice],
                                                             3, axes=axes)
        data_dict[Tags.DATA_FIELD_DENSITY] = np.rot90(tmp_ac_data[Tags.DATA_FIELD_DENSITY][image_slice],
                                                      3, axes=axes)
        data_dict[Tags.DATA_FIELD_ALPHA_COEFF] = np.rot90(tmp_ac_data[Tags.DATA_FIELD_ALPHA_COEFF][image_slice],
                                                          3, axes=axes)
        data_dict[Tags.DATA_FIELD_INITIAL_PRESSURE] = np.rot90(data_dict[Tags.DATA_FIELD_INITIAL_PRESSURE]
                                                                  [wavelength][image_slice], 3, axes=axes)

        time_series_data, global_settings = self.k_wave_acoustic_forward_model(
            detection_geometry,
            data_dict[Tags.DATA_FIELD_SPEED_OF_SOUND],
            data_dict[Tags.DATA_FIELD_DENSITY],
            data_dict[Tags.DATA_FIELD_ALPHA_COEFF],
            data_dict[Tags.DATA_FIELD_INITIAL_PRESSURE],
            optical_path=self.global_settings[Tags.SIMPA_OUTPUT_PATH])
        save_hdf5(global_settings, global_settings[Tags.SIMPA_OUTPUT_PATH], "/settings/")

        return time_series_data

    def k_wave_acoustic_forward_model(self, detection_geometry: DetectionGeometryBase,
                                      speed_of_sound: float, density: float,
                                      alpha_coeff: float, initial_pressure: np.ndarray,
                                      optical_path: str = "temporary") -> tuple:
        """
        Runs the acoustic forward model with the given parameters speed_of_sound (float), density (float),
        alpha_coeff (float) for the initial_pressure distribution (numpy array) and a given detection geometry.
        Uses the given optical_path (str) or if none is given a temporary one for saving temporary files.
        Note, that in order to work properly, this function assumes that several settings mentioned above are set.
        This can either be done by reading it from a settings file (e.g. when being called from forward_model) or
        by parsing all settings individually as in the convenience function
        (perform_k_wave_acoustic_forward_simulation).

        :param detection_geometry:
        :param speed_of_sound:
        :param density:
        :param alpha_coeff:
        :param initial_pressure:
        :param optical_path:
        :return: time_series_data (numpy array): simulated time series data, global_settings (Settings): updated global
            settings with new entries from the simulation

        """
        data_dict = {}

        pa_device = detection_geometry
        field_of_view = pa_device.get_field_of_view_mm()
        detector_positions_mm = pa_device.get_detector_element_positions_accounting_for_device_position_mm()

        if not self.component_settings.get(Tags.ACOUSTIC_SIMULATION_3D):
            detectors_are_aligned_along_x_axis = np.abs(field_of_view[2] - field_of_view[3]) < 1e-5
            detectors_are_aligned_along_y_axis = np.abs(field_of_view[0] - field_of_view[1]) < 1e-5
            if detectors_are_aligned_along_x_axis or detectors_are_aligned_along_y_axis:
                simulate_2d = True
            else:
                simulate_2d = False
        else:
            simulate_2d = False

        data_dict[Tags.DATA_FIELD_SPEED_OF_SOUND] = np.ones_like(initial_pressure) * speed_of_sound
        data_dict[Tags.DATA_FIELD_DENSITY] = np.ones_like(initial_pressure) * density
        data_dict[Tags.DATA_FIELD_ALPHA_COEFF] = np.ones_like(initial_pressure) * alpha_coeff
        data_dict[Tags.DATA_FIELD_INITIAL_PRESSURE] = initial_pressure

        if simulate_2d:
            detector_positions_mm_2d = np.delete(detector_positions_mm, 1, axis=1)
            detector_positions_mm_2d = np.moveaxis(detector_positions_mm_2d, 1, 0)
            data_dict[Tags.SENSOR_ELEMENT_POSITIONS] = detector_positions_mm_2d[[1, 0]]
            orientations = pa_device.get_detector_element_orientations()
            angles = np.arccos(np.dot(orientations, np.array([1, 0, 0])))
            data_dict[Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE] = angles[::-1]
        else:
            detector_positions_mm = np.moveaxis(detector_positions_mm, 1, 0)
            data_dict[Tags.SENSOR_ELEMENT_POSITIONS] = detector_positions_mm[[2, 1, 0]]
            orientations = pa_device.get_detector_element_orientations()
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
            data_dict[Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE] = angles
            data_dict[Tags.KWAVE_PROPERTY_INTRINSIC_EULER_ANGLE] = intrinsic_euler_angles

        optical_path = optical_path + ".mat"
        optical_path = os.path.abspath(optical_path)

        possible_k_wave_parameters = [Tags.SPACING_MM, Tags.MODEL_SENSOR_FREQUENCY_RESPONSE,
                                      Tags.KWAVE_PROPERTY_ALPHA_POWER, Tags.GPU, Tags.KWAVE_PROPERTY_PMLInside, Tags.KWAVE_PROPERTY_PMLAlpha, Tags.KWAVE_PROPERTY_PlotPML,
                                      Tags.RECORDMOVIE, Tags.MOVIENAME, Tags.ACOUSTIC_LOG_SCALE,
                                      Tags.SENSOR_DIRECTIVITY_PATTERN, Tags.KWAVE_PROPERTY_INITIAL_PRESSURE_SMOOTHING]

        k_wave_settings = Settings({
            Tags.SENSOR_NUM_ELEMENTS: pa_device.number_detector_elements,
            Tags.DETECTOR_ELEMENT_WIDTH_MM: pa_device.detector_element_width_mm,
            Tags.SENSOR_CENTER_FREQUENCY_HZ: pa_device.center_frequency_Hz,
            Tags.SENSOR_BANDWIDTH_PERCENT: pa_device.bandwidth_percent,
            Tags.SENSOR_SAMPLING_RATE_MHZ: pa_device.sampling_frequency_MHz
        })
        if isinstance(pa_device, CurvedArrayDetectionGeometry):
            k_wave_settings[Tags.SENSOR_RADIUS_MM] = pa_device.radius_mm

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

        del data_dict, k_wave_settings, detector_positions_mm, pa_device
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
        self.logger.info(cmd)
        subprocess.run(cmd)

        raw_time_series_data = sio.loadmat(optical_path)[Tags.DATA_FIELD_TIME_SERIES_DATA]

        # reverse the order of detector elements from matlab to python order
        raw_time_series_data = raw_time_series_data[::-1, :]

        time_grid = sio.loadmat(optical_path + "dt.mat")
        num_time_steps = int(np.round(time_grid["number_time_steps"]))

        self.global_settings[Tags.K_WAVE_SPECIFIC_DT] = float(time_grid["time_step"])
        self.global_settings[Tags.K_WAVE_SPECIFIC_NT] = num_time_steps

        os.remove(optical_path)
        os.remove(optical_path + "dt.mat")
        os.chdir(cur_dir)

        return raw_time_series_data, self.global_settings


def perform_k_wave_acoustic_forward_simulation(initial_pressure: np.array,
                                               detection_geometry: DetectionGeometryBase,
                                               speed_of_sound: float = 1540.0,
                                               density: float = 1000.0,
                                               alpha_coeff: float = 0.02,
                                               acoustic_settings: Settings = None,
                                               alpha_power: float = 0.0,
                                               sensor_record: str = "p",
                                               pml_inside: bool = False,
                                               pml_alpha: float = 1.5,
                                               plot_pml: bool = False,
                                               record_movie: bool = False,
                                               movie_name: str = "visualization_log",
                                               acoustic_log_scale: bool = True,
                                               gpu: bool = True,
                                               spacing_mm: float = 0.5) -> np.array:
    """
    Convenience function for performing a k-Wave acoustic forward simulation using a given detection geometry and
    initial pressure distribution (numpy array) with the following parameters speed_of_sound (float), density (float),
    alpha_coeff (float) as well as acoustic_settings (Settings). The acoustic settings may be parsed individually,
    however, they will be overwritten if they are also set in the acoustic_settings.

    :param initial_pressure:
    :param detection_geometry:
    :param speed_of_sound:
    :param density:
    :param alpha_coeff:
    :param acoustic_settings:
    :param alpha_power:
    :param sensor_record:
    :param pml_inside:
    :param pml_alpha:
    :param plot_pml:
    :param record_movie:
    :param movie_name:
    :param acoustic_log_scale:
    :param gpu:
    :param spacing_mm:
    :return: simulated time series data (numpy array)
    """

    # check and set acoustic settings
    if acoustic_settings is None:
        acoustic_settings = Settings()

    pm = PathManager()
    if Tags.ACOUSTIC_MODEL_BINARY_PATH not in acoustic_settings or \
            acoustic_settings[Tags.ACOUSTIC_MODEL_BINARY_PATH] is None:
        acoustic_settings[Tags.ACOUSTIC_MODEL_BINARY_PATH] = pm.get_matlab_binary_path()

    if Tags.KWAVE_PROPERTY_ALPHA_POWER not in acoustic_settings or acoustic_settings[Tags.KWAVE_PROPERTY_ALPHA_POWER] is None:
        acoustic_settings[Tags.KWAVE_PROPERTY_ALPHA_POWER] = alpha_power

    if Tags.KWAVE_PROPERTY_SENSOR_RECORD not in acoustic_settings or acoustic_settings[Tags.KWAVE_PROPERTY_SENSOR_RECORD] is None:
        acoustic_settings[Tags.KWAVE_PROPERTY_SENSOR_RECORD] = sensor_record

    if Tags.KWAVE_PROPERTY_PMLInside not in acoustic_settings or acoustic_settings[Tags.KWAVE_PROPERTY_PMLInside] is None:
        acoustic_settings[Tags.KWAVE_PROPERTY_PMLInside] = pml_inside

    if Tags.KWAVE_PROPERTY_PMLAlpha not in acoustic_settings or acoustic_settings[Tags.KWAVE_PROPERTY_PMLAlpha] is None:
        acoustic_settings[Tags.KWAVE_PROPERTY_PMLAlpha] = pml_alpha

    if Tags.KWAVE_PROPERTY_PlotPML not in acoustic_settings or acoustic_settings[Tags.KWAVE_PROPERTY_PlotPML] is None:
        acoustic_settings[Tags.KWAVE_PROPERTY_PlotPML] = plot_pml

    if Tags.RECORDMOVIE not in acoustic_settings or acoustic_settings[Tags.RECORDMOVIE] is None:
        acoustic_settings[Tags.RECORDMOVIE] = record_movie

    if Tags.MOVIENAME not in acoustic_settings or acoustic_settings[Tags.MOVIENAME] is None:
        acoustic_settings[Tags.MOVIENAME] = movie_name

    if Tags.ACOUSTIC_LOG_SCALE not in acoustic_settings or acoustic_settings[Tags.ACOUSTIC_LOG_SCALE] is None:
        acoustic_settings[Tags.ACOUSTIC_LOG_SCALE] = acoustic_log_scale

    if Tags.GPU not in acoustic_settings or acoustic_settings[Tags.GPU] is None:
        acoustic_settings[Tags.GPU] = gpu

    if Tags.SPACING_MM not in acoustic_settings or acoustic_settings[Tags.SPACING_MM] is None:
        acoustic_settings[Tags.SPACING_MM] = spacing_mm

    settings = Settings()
    settings.set_acoustic_settings(acoustic_settings)

    # initialize adapter and run forward model
    kWave = KWaveAdapter(settings)
    time_series_data, updated_global_settings = kWave.k_wave_acoustic_forward_model(
        detection_geometry, speed_of_sound, density, alpha_coeff, initial_pressure)
    return time_series_data
