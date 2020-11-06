# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


class Tags:
    """
    This class contains all 'Tags' for the use in the settings dictionary.
    """



    """
    General settings
    """
    SIMULATION_PATH = ("simulation_path", str)
    VOLUME_NAME = ("volume_name", str)
    WAVELENGTHS = ("wavelengths", (list, range, tuple, np.ndarray))
    WAVELENGTH = ("wavelength", (int, np.integer))
    RANDOM_SEED = ("random_seed", (int, np.integer))
    TISSUE_PROPERTIES_OUPUT_NAME = "properties"
    SIMULATION_EXTRACT_FIELD_OF_VIEW = ("extract_field_of_view", bool)
    GPU = ("gpu", bool)
    ACOUSTIC_SIMULATION_3D = ("acoustic_simulation_3d", bool)
    MEDIUM_TEMPERATURE_CELCIUS = ("medium_temperature", (int, np.integer, float, np.float))

    """
    Volume Creation Settings
    """
    VOLUME_CREATOR = ("volume_creator", str)
    VOLUME_CREATOR_VERSATILE = "volume_creator_versatile"
    PRIORITY = "priority"
    MOLECULE_COMPOSITION = "molecule_composition"

    """
    Optical model settings
    """
    RUN_OPTICAL_MODEL = ("run_optical_forward_model", bool)
    OPTICAL_MODEL_OUTPUT_NAME = "optical_forward_model_output"
    OPTICAL_MODEL_BINARY_PATH = ("optical_model_binary_path", str)
    OPTICAL_MODEL_NUMBER_PHOTONS = ("optical_model_number_of_photons", (int, np.integer, float, np.float))
    OPTICAL_MODEL_ILLUMINATION_GEOMETRY_XML_FILE = ("optical_model_illumination_geometry_xml_file", str)
    LASER_PULSE_ENERGY_IN_MILLIJOULE = ("laser_pulse_energy_in_millijoule", (int, np.integer, float, np.float, list, range, tuple, np.ndarray))
    OPTICAL_MODEL_FLUENCE = "fluence"
    OPTICAL_MODEL_INITIAL_PRESSURE = "initial_pressure"
    OPTICAL_MODEL_UNITS = "units"

    ILLUMINATION_TYPE = ("optical_model_illumination_type", str)

    # Illumination parameters
    ILLUMINATION_POSITION = ("illumination_position", (list, tuple, np.ndarray))
    ILLUMINATION_DIRECTION = ("illumination_direction", (list, tuple, np.ndarray))
    ILLUMINATION_PARAM1 = ("illumination_param1", (list, tuple, np.ndarray))
    ILLUMINATION_PARAM2 = ("illumination_param2", (list, tuple, np.ndarray))
    TIME_STEP = ("time_step", (int, np.integer, float, np.float))
    TOTAL_TIME = ("total_time", (int, np.integer, float, np.float))

    # Supported illumination types - implemented in mcx
    ILLUMINATION_TYPE_PENCIL = "pencil"
    ILLUMINATION_TYPE_DISK = "disk"
    ILLUMINATION_TYPE_SLIT = "slit"
    ILLUMINATION_TYPE_GAUSSIAN = "gaussian"
    ILLUMINATION_TYPE_PATTERN = "pattern"
    ILLUMINATION_TYPE_PATTERN_3D = "pattern3d"
    ILLUMINATION_TYPE_FOURIER = "fourier"
    ILLUMINATION_TYPE_FOURIER_X = "fourierx"
    ILLUMINATION_TYPE_FOURIER_X_2D = "fourierx2d"

    ILLUMINATION_TYPE_DKFZ_PAUS = "pasetup"  # TODO more explanatory rename of pasetup
    ILLUMINATION_TYPE_MSOT_ACUITY_ECHO = "msot_acuity_echo"

    # Supported optical models
    OPTICAL_MODEL = ("optical_model", str)
    OPTICAL_MODEL_MCXYZ = "mcxyz"
    OPTICAL_MODEL_MCX = "mcx"
    OPTICAL_MODEL_TEST = "simpa_tests"

    # Supported acoustic models
    ACOUSTIC_MODEL = ("acoustic_model", str)
    ACOUSTIC_MODEL_K_WAVE = "kwave"
    ACOUSTIC_MODEL_TEST = "simpa_tests"
    ACOUSTIC_MODEL_SCRIPT = "acoustic_model_script"
    ACOUSTIC_MODEL_SCRIPT_LOCATION = ("acoustic_model_script_location", str)

    """
    Acoustic model settings
    """

    RUN_ACOUSTIC_MODEL = ("run_acoustic_forward_model", bool)
    ACOUSTIC_MODEL_BINARY_PATH = ("acoustic_model_binary_path", str)
    ACOUSTIC_MODEL_OUTPUT_NAME = "acoustic_forward_model_output"
    ACOUSTIC_SIMULATION_PATH = "acoustic_simulation_path"
    RECORDMOVIE = ("record_movie", bool)
    MOVIENAME = "movie_name"
    ACOUSTIC_PLOT_SCALE = "acoustic_plot_scale"
    ACOUSTIC_LOG_SCALE = "acoustic_log_scale"
    TIME_SERIES_DATA = "time_series_data"
    TIME_SERIES_DATA_NOISE = "time_series_data_noise"

    # Reconstruction settings
    PERFORM_IMAGE_RECONSTRUCTION = ("perform_image_reconstruction", bool)
    RECONSTRUCTION_OUTPUT_NAME = "reconstruction_result"
    RECONSTRUCTION_ALGORITHM = ("reconstruction_algorithm", str)
    RECONSTRUCTION_ALGORITHM_DAS = "DAS"
    RECONSTRUCTION_ALGORITHM_DMAS = "DMAS"
    RECONSTRUCTION_ALGORITHM_SDMAS = "sDMAS"
    RECONSTRUCTION_ALGORITHM_TIME_REVERSAL = "time_reversal"
    RECONSTRUCTION_ALGORITHM_TEST = "TEST"
    RECONSTRUCTION_INVERSE_CRIME = ("reconstruction_inverse_crime", bool)
    RECONSTRUCTION_MITK_BINARY_PATH = ("reconstruction_mitk_binary_path", str)
    RECONSTRUCTION_MITK_SETTINGS_XML = ("reconstruction_mitk_settings_xml", str)
    RECONSTRUCTION_BMODE_METHOD = "reconstruction_bmode_method"
    RECONSTRUCTION_BMODE_METHOD_ABS = "Abs"
    RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM = "EnvelopeDetection"
    RECONSTRUCTED_DATA = "reconstructed_data"
    RECONSTRUCTED_DATA_NOISE = "reconstructed_data_noise"

    """
    Upsampling settings
    """

    CROP_IMAGE = ("crop_image", bool)
    CENTER_CROP = "center_crop"
    CROP_POWER_OF_TWO = "crop_power_of_two"
    PERFORM_UPSAMPLING = ("sample", bool)
    UPSAMPLING_METHOD = "upsampling_method"
    UPSAMPLING_METHOD_DEEP_LEARNING = "deeplearning"
    UPSAMPLING_METHOD_NEAREST_NEIGHBOUR = "nearestneighbour"
    UPSAMPLING_METHOD_BILINEAR = "bilinear"
    UPSAMPLING_METHOD_LANCZOS2 = "lanczos2"
    UPSAMPLING_METHOD_LANCZOS3 = "lanczos3"
    UPSAMPLING_SCRIPT = ("upsampling_script", str)
    UPSAMPLING_SCRIPT_LOCATION = "upsampling_script_location"
    UPSCALE_FACTOR = "upscale_factor"
    UPSAMPLING_RUN = ("upsampling_run", bool)
    DL_MODEL_PATH = "dl_model_path"

    # physical property volume types
    PROPERTY_ABSORPTION_PER_CM = "mua"
    PROPERTY_SCATTERING_PER_CM = "mus"
    PROPERTY_ANISOTROPY = "g"
    PROPERTY_OXYGENATION = "oxy"
    PROPERTY_SEGMENTATION = "seg"
    """
    We define PROPERTY_GRUNEISEN_PARAMETER to contain all wavelength-independent constituents of the PA signal.
    This means that it contains the percentage of absorbed light converted into heat.
    Naturally, one could make an argument that this should not be the case, however, it simplifies the usage of 
    this tool.
    """
    PROPERTY_GRUNEISEN_PARAMETER = "gamma"
    PROPERTY_SPEED_OF_SOUND = "sos"
    PROPERTY_DENSITY = "density"
    PROPERTY_ALPHA_COEFF = "alpha_coeff"
    PROPERTY_SENSOR_MASK = "sensor_mask"
    PROPERTY_DIRECTIVITY_ANGLE = "directivity_angle"

    # Air layer
    AIR_LAYER = ("airlayer", bool)
    AIR_LAYER_HEIGHT_MM = ("air_layer_height", (int, np.integer, float, np.float))

    # Gel Pad Layer
    GELPAD_LAYER = ("gelpad", bool)
    GELPAD_LAYER_HEIGHT_MM = ("gelpad_layer_height_mm", (int, np.integer, float, np.float))

    # Volume geometry settings
    SPACING_MM = ("voxel_spacing_mm", (int, np.integer, float, np.float))
    DIM_VOLUME_X_MM = ("volume_x_dim_mm", (int, np.integer, float, np.float))
    DIM_VOLUME_Y_MM = ("volume_y_dim_mm", (int, np.integer, float, np.float))
    DIM_VOLUME_Z_MM = ("volume_z_dim_mm", (int, np.integer, float, np.float))

    # 2D Acoustic Medium Properties
    MEDIUM_SOUND_SPEED_HOMOGENEOUS = ("medium_sound_speed_homogeneous", bool)
    MEDIUM_SOUND_SPEED = "medium_sound_speed"
    MEDIUM_DENSITY_HOMOGENEOUS = "medium_density_homogeneous"
    MEDIUM_DENSITY = "medium_density"
    MEDIUM_ALPHA_COEFF_HOMOGENEOUS = "medium_alpha_coeff_homogeneous"
    MEDIUM_ALPHA_COEFF = "medium_alpha_coeff"
    MEDIUM_ALPHA_POWER = "medium_alpha_power"
    MEDIUM_NONLINEARITY = "medium_nonlinearity"

    # PML parameters

    PMLSize = "pml_size"
    PMLAlpha = "pml_alpha"
    PMLInside = "pml_inside"
    PlotPML = "plot_pml"

    # Acoustic Sensor Properties
    SENSOR_MASK = "sensor_mask"
    SENSOR_RECORD = "sensor_record"
    SENSOR_CENTER_FREQUENCY_HZ = "sensor_center_frequency"
    SENSOR_BANDWIDTH_PERCENT = "sensor_bandwidth"
    SENSOR_DIRECTIVITY_HOMOGENEOUS = "sensor_directivity_homogeneous"
    SENSOR_DIRECTIVITY_ANGLE = "sensor_directivity_angle"
    SENSOR_DIRECTIVITY_SIZE_M = "sensor_directivity_size"
    SENSOR_DIRECTIVITY_PATTERN = "sensor_directivity_pattern"
    SENSOR_ELEMENT_PITCH_MM = "sensor_element_pitch"
    SENSOR_SAMPLING_RATE_MHZ = "sensor_sampling_rate_mhz"
    SENSOR_NUM_ELEMENTS = "sensor_num_elements"
    SENSOR_NUM_USED_ELEMENTS = "sensor_num_used_elements"
    SENSOR_CONCAVE = "concave"
    SENSOR_RADIUS_MM = "sensor_radius_mm"
    SENSOR_LINEAR = "linear"

    # Noise properties
    APPLY_NOISE_MODEL = ("apply_noise_model", bool)
    NOISE_MODEL = "noise_model"
    NOISE_MODEL_GAUSSIAN = "noise_model_gaussian"
    NOISE_MEAN = "noise_mean"
    NOISE_STD = "noise_std"
    NOISE_MODEL_OUTPUT_NAME = "noise_model_output"
    NOISE_MODEL_PATH = "noise_model_path"

    # Constant Tissue Properties
    KEY_CONSTANT_PROPERTIES = "constant_properties"
    KEY_MUA = "mua"
    KEY_MUS = "mus"
    KEY_G = "g"

    # Tissue Properties Settings
    KEY_B = "B"
    KEY_B_MIN = "B_min"
    KEY_B_MAX = "B_max"
    KEY_W = "W"
    KEY_W_MAX = "w_max"
    KEY_W_MIN = "w_min"
    KEY_F = "F"
    KEY_F_MAX = "f_max"
    KEY_F_MIN = "f_min"
    KEY_M = "M"
    KEY_M_MAX = "m_max"
    KEY_M_MIN = "m_min"
    KEY_OXY = "OXY"
    KEY_OXY_MAX = "oxy_max"
    KEY_OXY_MIN = "oxy_min"
    KEY_MUSP500 = "musp500"
    KEY_F_RAY = "f_ray"
    KEY_B_MIE = "b_mie"
    KEY_ANISOTROPY = "anisotropy"

    # Structures
    STRUCTURES = ("structures", dict)
    CHILD_STRUCTURES = "child_structures"
    STRUCTURE_TYPE = "structure_type"
    STRUCTURE_SEGMENTATION_TYPE = "structure_segmentation_type"
    STRUCTURE_TISSUE_PROPERTIES = "structure_tissue_properties"

    STRUCTURE_CENTER_DEPTH_MIN_MM = "structure_depth_min_mm"
    STRUCTURE_CENTER_DEPTH_MAX_MM = "structure_depth_max_mm"

    STRUCTURE_BACKGROUND = "structure_background"

    STRUCTURE_LAYER = "structure_layer"
    STRUCTURE_THICKNESS_MIN_MM = "structure_thickness_min_mm"
    STRUCTURE_THICKNESS_MAX_MM = "structure_thickness_max_mm"

    STRUCTURE_TUBE = "structure_tube"
    STRUCTURE_RADIUS_MIN_MM = "structure_radius_min_mm"
    STRUCTURE_RADIUS_MAX_MM = "structure_radius_max_mm"
    STRUCTURE_FORCE_ORTHOGONAL_TO_PLANE = "structure_force_orthogonal_to_plane"
    STRUCTURE_TUBE_CENTER_X_MIN_MM = "structure_tube_start_x_min_mm"
    STRUCTURE_TUBE_CENTER_X_MAX_MM = "structure_tube_start_x_max_mm"

    STRUCTURE_ELLIPSE = "structure_ellipse"
    STRUCTURE_MIN_ECCENTRICITY = "structure_eccentricity_min"
    STRUCTURE_MAX_ECCENTRICITY = "structure_eccentricity_max"

    STRUCTURE_DISTORTED_LAYERS = "distorted_layers"
    STRUCTURE_DISTORTED_LAYERS_ELEVATION = "distorted_layers_elevation"

    UNITS_ARBITRARY = "arbitrary_unity"
    UNITS_PRESSURE = "newton_per_meters_squared"

    """
    IO settings
    """

    SIMPA_OUTPUT_PATH = ("simpa_output_path", str)
    SIMPA_OUTPUT_NAME = "simpa_output.hdf5"
    SETTINGS_JSON = "settings_json"
    SETTINGS_JSON_PATH = "settings_json_path"
    SETTINGS = "settings"
    SIMULATION_PROPERTIES = "simulation_properties"
    SIMULATIONS = "simulations"
    UPSAMPLED_DATA = "upsampled_data"
    ORIGINAL_DATA = "original_data"
