# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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
    This class contains all 'Tags' for the use in the settings dictionary as well as strings that are used in SIMPA
    as naming conventions.
    Every Tag that is intended to be used as a key in the settings dictionary is represented by a tuple.
    The first element of the tuple is a string that corresponds to the name of the Tag.
    The second element of the tuple is a data type or a tuple of data types.
    The values that are assigned to the keys in the settings should match these data types.
    Their usage within the SIMPA package is divided in "SIMPA package", "module X", "adapter Y", "class Z" and
    "naming convention".
    """

    """
    General settings
    """

    SIMULATION_PATH = ("simulation_path", str)
    """
    Absolute path to the folder where the SIMPA output is saved.\n
    Usage: SIMPA package
    """

    VOLUME_NAME = ("volume_name", str)
    """
    Name of the SIMPA output file.\n
    Usage: SIMPA package
    """

    WAVELENGTHS = ("wavelengths", (list, range, tuple, np.ndarray))
    """
    Iterable of all the wavelengths used for the simulation.\n
    Usage: SIMPA package
    """

    WAVELENGTH = ("wavelength", (int, np.integer))
    """
    Single wavelength used for the current simulation.\n
    Usage: SIMPA package
    """

    RANDOM_SEED = ("random_seed", (int, np.integer))
    """
    Random seed for numpy and torch.\n
    Usage: SIMPA package
    """

    TISSUE_PROPERTIES_OUPUT_NAME = "properties"
    """
    Name of the simulation properties field in the SIMPA output file.\n
    Usage: naming convention
    """

    SIMULATION_EXTRACT_FIELD_OF_VIEW = ("extract_field_of_view", bool)
    """
    If True, converts a 3D volume to a 2D volume by extracting the middle slice along the y-axis.\n
    Usage: SIMPA package
    """

    GPU = ("gpu", (bool, np.bool, np.bool_))
    """
    If True, uses all available gpu options of the used modules.\n
    Usage: SIMPA package 
    """

    ACOUSTIC_SIMULATION_3D = ("acoustic_simulation_3d", bool)
    """
    If True, simulates the acoustic forward model in 3D.\n
    Usage: SIMPA package
    """

    MEDIUM_TEMPERATURE_CELCIUS = ("medium_temperature", (int, np.integer, float, np.float))
    """
    Temperature of the simulated volume.\n
    Usage: module noise_simulation
    """

    """
    Volume Creation Settings
    """

    VOLUME_CREATOR = ("volume_creator", str)
    """
    Choice of the volume creator adapter.\n 
    Usage: module volume_creation, module device_digital_twins
    """

    VOLUME_CREATOR_VERSATILE = "volume_creator_versatile"
    """
    Corresponds to the ModelBasedVolumeCreator.\n
    Usage: module volume_creation, naming convention
    """

    VOLUME_CREATOR_SEGMENTATION_BASED = "volume_creator_segmentation_based"
    """
    Corresponds to the SegmentationBasedVolumeCreator.\n
    Usage: module volume_creation, naming convention
    """

    INPUT_SEGMENTATION_VOLUME = ("input_segmentation_volume", np.ndarray)
    """
    Array that defines a segmented volume.\n
    Usage: adapter segmentation_based_volume_creator
    """

    SEGMENTATION_CLASS_MAPPING = ("segmentation_class_mapping", dict)
    """
    Mapping that assigns every class in the INPUT_SEGMENTATION_VOLUME a MOLECULE_COMPOSITION.\n
    Usage: adapter segmentation_based_volume_creator
    """

    PRIORITY = ("priority", (int, np.integer, float, np.float))
    """
    Number that corresponds to a priority of the assigned structure. If another structure occupies the same voxel 
    in a volume, the structure with a higher priority will be preferred.\n
    Usage: adapter versatile_volume_creator
    """

    MOLECULE_COMPOSITION = ("molecule_composition", list)
    """
    List that contains all the molecules within a structure.\n
    Usage: module volume_creation
    """

    SIMULATE_DEFORMED_LAYERS = ("simulate_deformed_layers", bool)
    """
    If True, the horizontal layers are deformed according to the DEFORMED_LAYERS_SETTINGS.\n
    Usage: adapter versatile_volume_creation
    """

    DEFORMED_LAYERS_SETTINGS = ("deformed_layers_settings", dict)
    """
    Settings that contain the functional which defines the deformation of the layers.\n
    Usage: adapter versatile_volume_creation
    """

    BACKGROUND = "Background"
    """
    Corresponds to the name of a structure.\n
    Usage: adapter versatile_volume_creation, naming convention
    """

    ADHERE_TO_DEFORMATION = ("adhere_to_deformation", bool)
    """
    If True, a structure will be shifted according to the deformation.\n
    Usage: adapter versatile_volume_creation
    """

    DEFORMATION_X_COORDINATES_MM = "deformation_x_coordinates"
    """
    Mesh that defines the x coordinates of the deformation.\n
    Usage: adapter versatile_volume_creation, naming convention
    """

    DEFORMATION_Y_COORDINATES_MM = "deformation_y_coordinates"
    """
    Mesh that defines the y coordinates of the deformation.\n
    Usage: adapter versatile_volume_creation, naming convention
    """

    DEFORMATION_Z_ELEVATIONS_MM = "deformation_z_elevation"
    """
    Mesh that defines the z coordinates of the deformation.\n
    Usage: adapter versatile_volume_creation, naming convention
    """

    MAX_DEFORMATION_MM = "max_deformation"
    """
    Maximum deformation in z-direction.\n
    Usage: adapter versatile_volume_creation, naming convention
    """

    """
    Structure Settings
    """

    CONSIDER_PARTIAL_VOLUME = ("consider_partial_volume", bool)
    """
    If True, the structure will be generated with its edges only occupying a partial volume of the voxel.\n
    Usage: adapter versatile_volume_creation
    """

    STRUCTURE_START_MM = ("structure_start", (list, tuple, np.ndarray))
    """
    Beginning of the structure as [x, y, z] coordinates in the generated volume.\n
    Usage: adapter versatile_volume_creation, class GeometricalStructure
    """

    STRUCTURE_END_MM = ("structure_end", (list, tuple, np.ndarray))
    """
    Ending of the structure as [x, y, z] coordinates in the generated volume.\n
    Usage: adapter versatile_volume_creation, class GeometricalStructure
    """

    STRUCTURE_RADIUS_MM = ("structure_radius", (int, np.integer, float, np.float, np.ndarray))
    """
    Radius of the structure.\n
    Usage: adapter versatile_volume_creation, class GeometricalStructure
    """

    STRUCTURE_ECCENTRICITY = ("structure_excentricity", (int, np.integer, float, np.float, np.ndarray))
    """
    Eccentricity of the structure.\n
    Usage: adapter versatile_volume_creation, class EllipticalTubularStructure
    """

    STRUCTURE_FIRST_EDGE_MM = ("structure_first_edge_mm", (list, tuple, np.ndarray))
    """
    Edge of the structure as [x, y, z] vector starting from STRUCTURE_START_MM in the generated volume.\n
    Usage: adapter versatile_volume_creation, class ParallelepipedStructure
    """

    STRUCTURE_SECOND_EDGE_MM = ("structure_second_edge_mm", (list, tuple, np.ndarray))
    """
    Edge of the structure as [x, y, z] vector starting from STRUCTURE_START_MM in the generated volume.\n
    Usage: adapter versatile_volume_creation, class ParallelepipedStructure
    """

    STRUCTURE_THIRD_EDGE_MM = ("structure_third_edge_mm", (list, tuple, np.ndarray))
    """
    Edge of the structure as [x, y, z] vector starting from STRUCTURE_START_MM in the generated volume.\n
    Usage: adapter versatile_volume_creation, class ParallelepipedStructure
    """

    STRUCTURE_X_EXTENT_MM = ("structure_x_extent_mm", (int, np.integer, float, np.float))
    """
    X-extent of the structure in the generated volume.\n
    Usage: adapter versatile_volume_creation, class RectangularCuboidStructure
    """

    STRUCTURE_Y_EXTENT_MM = ("structure_y_extent_mm", (int, np.integer, float, np.float))
    """
    Y-extent of the structure in the generated volume.\n
    Usage: adapter versatile_volume_creation, class RectangularCuboidStructure
    """

    STRUCTURE_Z_EXTENT_MM = ("structure_z_extent_mm", (int, np.integer, float, np.float))
    """
    Z-extent of the structure in the generated volume.\n
    Usage: adapter versatile_volume_creation, class RectangularCuboidStructure
    """

    STRUCTURE_BIFURCATION_LENGTH_MM = ("structure_bifurcation_length_mm", (int, np.integer, float, np.float))
    """
    Length after which a VesselStructure will bifurcate.\n
    Usage: adapter versatile_volume_creation, class VesselStructure
    """

    STRUCTURE_CURVATURE_FACTOR = ("structure_curvature_factor", (int, np.integer, float, np.float))
    """
    Factor that determines how strongly a vessel tree is curved.\n
    Usage: adapter versatile_volume_creation, class VesselStructure
    """

    STRUCTURE_RADIUS_VARIATION_FACTOR = ("structure_radius_variation_factor", (int, np.integer, float, np.float))
    """
    Factor that determines how strongly a the radius of vessel tree varies.\n
    Usage: adapter versatile_volume_creation, class VesselStructure
    """

    STRUCTURE_DIRECTION = ("structure_direction", (list, tuple, np.ndarray))
    """
    Direction as [x, y, z] vector starting from STRUCTURE_START_MM in which the vessel will grow.\n
    Usage: adapter versatile_volume_creation, class VesselStructure
    """

    """
    Digital Device Twin Settings
    """

    DIGITAL_DEVICE = ("digital_device", str)
    """
    Digital device that is chosen as illumination source and detector for the simulation.\n
    Usage: SIMPA package
    """

    DIGITAL_DEVICE_MSOT = "digital_device_msot"
    """
    Corresponds to the MSOTAcuityEcho device.\n
    Usage: SIMPA package, naming convention
    """

    DIGITAL_DEVICE_RSOM = "digital_device_rsom"
    """
    Corresponds to the RSOMExplorerP50 device.\n
    Usage: SIMPA package, naming convention
    """

    DIGITAL_DEVICE_INVISION = "digital_device_invision"
    """
    Corresponds to the InVision 256-TF device.\n
    Usage: SIMPA package, naming convention
    """

    DIGITAL_DEVICE_POSITION = ("digital_device_position", (list, tuple, np.ndarray))
    """
    Position in [x, y, z] coordinates of the device in the generated volume.\n
    Usage: SIMPA package
    """

    US_GEL = ("us_gel", bool)
    """
    If True, us gel is placed between the PA device and the simulated volume.\n
    Usage: SIMPA package
    """

    """
    Optical model settings
    """

    RUN_OPTICAL_MODEL = ("run_optical_forward_model", bool)
    """
    If True, the simulation will run the optical forward model.\n
    Usage: module core (simulate.py)
    """

    OPTICAL_MODEL_OUTPUT_NAME = "optical_forward_model_output"
    """
    Name of the optical forward model output field in the SIMPA output file.\n
    Usage: naming convention
    """

    OPTICAL_MODEL_BINARY_PATH = ("optical_model_binary_path", str)
    """
    Absolute path of the location of the optical forward model binary.\n
    Usage: module optical_simulation
    """

    OPTICAL_MODEL_NUMBER_PHOTONS = ("optical_model_number_of_photons", (int, np.integer, float, np.float))
    """
    Number of photons used in the optical simulation.\n
    Usage: module optical_simulation
    """

    OPTICAL_MODEL_ILLUMINATION_GEOMETRY_XML_FILE = ("optical_model_illumination_geometry_xml_file", str)
    """
    Absolute path of the location of the optical forward model illumination geometry.\n
    Usage: module optical_simulation
    """

    LASER_PULSE_ENERGY_IN_MILLIJOULE = ("laser_pulse_energy_in_millijoule", (int, np.integer, float, np.float, list,
                                                                             range, tuple, np.ndarray))
    """
    Laser pulse energy used in the optical simulation.\n
    Usage: module optical_simulation
    """

    OPTICAL_MODEL_FLUENCE = "fluence"
    """
    Name of the optical forward model output fluence field in the SIMPA output file.\n
    Usage: naming convention
    """

    OPTICAL_MODEL_INITIAL_PRESSURE = "initial_pressure"
    """
    Name of the optical forward model output initial pressure field in the SIMPA output file.\n
    Usage: naming convention
    """

    OPTICAL_MODEL_UNITS = "units"
    """
    Name of the optical forward model output units field in the SIMPA output file.\n
    Usage: naming convention
    """

    ILLUMINATION_TYPE = ("optical_model_illumination_type", str)
    """
    Type of the illumination geometry used in mcx.\n
    Usage: module optical_modelling, adapter mcx_adapter
    """

    # Illumination parameters
    ILLUMINATION_POSITION = ("illumination_position", (list, tuple, np.ndarray))
    """
    Position of the photon source in [x, y, z] coordinates used in mcx.\n
    Usage: module optical_modelling, adapter mcx_adapter
    """

    ILLUMINATION_DIRECTION = ("illumination_direction", (list, tuple, np.ndarray))
    """
    Direction of the photon source as [x, y, z] vector used in mcx.\n
    Usage: module optical_modelling, adapter mcx_adapter
    """

    ILLUMINATION_PARAM1 = ("illumination_param1", (list, tuple, np.ndarray))
    """
    First parameter group of the specified illumination type as [x, y, z, w] vector used in mcx.\n
    Usage: module optical_modelling, adapter mcx_adapter
    """

    ILLUMINATION_PARAM2 = ("illumination_param2", (list, tuple, np.ndarray))
    """
    Second parameter group of the specified illumination type as [x, y, z, w] vector used in mcx.\n
    Usage: module optical_modelling, adapter mcx_adapter
    """

    TIME_STEP = ("time_step", (int, np.integer, float, np.float))
    """
    Temporal resolution of mcx.\n
    Usage: adapter mcx_adapter
    """

    TOTAL_TIME = ("total_time", (int, np.integer, float, np.float))
    """
    Total simulated time in mcx.\n
    Usage: adapter mcx_adapter
    """

    # Supported illumination types - implemented in mcx
    ILLUMINATION_TYPE_PENCIL = "pencil"
    """
    Corresponds to pencil source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_PENCILARRAY = "pencilarray"
    """
    Corresponds to pencilarray source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_DISK = "disk"
    """
    Corresponds to disk source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_SLIT = "slit"
    """
    Corresponds to slit source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_GAUSSIAN = "gaussian"
    """
    Corresponds to gaussian source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_PATTERN = "pattern"
    """
    Corresponds to pattern source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_PATTERN_3D = "pattern3d"
    """
    Corresponds to pattern3d source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_PLANAR = "planar"
    """
    Corresponds to planar source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_FOURIER = "fourier"
    """
    Corresponds to fourier source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_FOURIER_X = "fourierx"
    """
    Corresponds to fourierx source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_FOURIER_X_2D = "fourierx2d"
    """
    Corresponds to fourierx2d source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_DKFZ_PAUS = "pasetup"  # TODO more explanatory rename of pasetup
    """
    Corresponds to pasetup source in mcx. The geometrical definition is described in:\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_MSOT_ACUITY_ECHO = "msot_acuity_echo"
    """
    Corresponds to msot_acuity_echo source in mcx. The device is manufactured by iThera Medical, Munich, Germany
    (https: // www.ithera-medical.com / products / msot-acuity /).\n
    Usage: adapter mcx_adapter, naming convention
    """

    ILLUMINATION_TYPE_RING = "ring"
    """
    Corresponds to ring source in mcx.\n
    Usage: adapter mcx_adapter, naming convention
    """

    # Supported optical models
    OPTICAL_MODEL = ("optical_model", str)
    """
    Choice of the used optical model.\n
    Usage: module optical_simulation
    """

    OPTICAL_MODEL_MCXYZ = "mcxyz"
    """
    Corresponds to the mcxyz simulation.\n
    Usage: module optical_simulation, naming convention
    """

    OPTICAL_MODEL_MCX = "mcx"
    """
    Corresponds to the mcx simulation.\n
    Usage: module optical_simulation, naming convention
    """

    OPTICAL_MODEL_TEST = "simpa_tests"
    """
    Corresponds to an adapter for testing purposes only.\n
    Usage: module optical_simulation, naming convention
    """

    # Supported acoustic models
    ACOUSTIC_MODEL = ("acoustic_model", str)
    """
    Choice of the used acoustic model.\n
    Usage: module acoustic_simulation
    """

    ACOUSTIC_MODEL_K_WAVE = "kwave"
    """
    Corresponds to the kwave simulaiton.\n
    Usage: module acoustic_simulation, naming convention
    """

    K_WAVE_SPECIFIC_DT = ("dt_acoustic_sim", (int, np.integer, float, np.float))
    """
    Temporal resolution of kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter
    """

    K_WAVE_SPECIFIC_NT = ("Nt_acoustic_sim", (int, np.integer, float, np.float))
    """
    Total time steps simulated by kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter
    """

    ACOUSTIC_MODEL_TEST = "simpa_tests"
    """
    Corresponds to an adapter for testing purposes only.\n
    Usage: module acoustic_simulation, naming convention
    """

    ACOUSTIC_MODEL_SCRIPT_LOCATION = ("acoustic_model_script_location", str)
    """
    Absolute path of the location of the acoustic_simulation folder in the SIMPA core module.\n
    Usage: module acoustic_simulation
    """

    TIME_REVEARSAL_SCRIPT_LOCATION = ("time_revearsal_script_location", str)
    """
    Absolute path of the location of the image_reconstruction folder in the SIMPA core module.\n
    Usage: adapter TimeReversalAdapter
    """

    """
    Acoustic model settings
    """

    RUN_ACOUSTIC_MODEL = ("run_acoustic_forward_model", (bool, np.bool, np.bool_))
    """
    If True, the simulation will run the acoustic forward model.\n
    Usage: module core (simulate.py)
    """

    ACOUSTIC_MODEL_BINARY_PATH = ("acoustic_model_binary_path", str)
    """
    Absolute path of the location of the acoustic forward model binary.\n
    Usage: module optical_simulation
    """

    ACOUSTIC_MODEL_OUTPUT_NAME = "acoustic_forward_model_output"
    """
    Name of the acoustic forward model output field in the SIMPA output file.\n
    Usage: naming convention
    """

    RECORDMOVIE = ("record_movie", (bool, np.bool, np.bool_))
    """
    If True, a movie of the kwave simulation will be recorded.\n
    Usage: adapter KwaveAcousticForwardModel
    """

    MOVIENAME = ("movie_name", str)
    """
    Name of the movie recorded by kwave.\n
    Usage: adapter KwaveAcousticForwardModel
    """

    ACOUSTIC_LOG_SCALE = ("acoustic_log_scale", (bool, np.bool, np.bool_))
    """
    If True, the movie of the kwave simulation will be recorded in a log scale.\n
    Usage: adapter KwaveAcousticForwardModel
    """

    TIME_SERIES_DATA = "time_series_data"
    """
    Name of the time series data field in the SIMPA output file.\n
    Usage: naming convention
    """

    TIME_SERIES_DATA_NOISE = "time_series_data_noise"
    """
    Name of the time series data with applied noise field in the SIMPA output file.\n
    Usage: naming convention
    """

    # Reconstruction settings
    PERFORM_IMAGE_RECONSTRUCTION = ("perform_image_reconstruction", (bool, np.bool, np.bool_))
    """
    If True, the simulation will run the image reconstruction.\n
    Usage: module core (simulate.py)
    """

    RECONSTRUCTION_OUTPUT_NAME = ("reconstruction_result", str)
    """
    Absolute path of the image reconstruction result.\n
    Usage: adapter MitkBeamformingAdapter
    """

    RECONSTRUCTION_ALGORITHM = ("reconstruction_algorithm", str)
    """
    Choice of the used reconstruction algorithm.\n
    Usage: module image_reconstruction
    """

    RECONSTRUCTION_ALGORITHM_DAS = "DAS"
    """
    Corresponds to the reconstruction algorithm DAS with the MitkBeamformingAdapter.\n
    Usage: module image_reconstruction, naming convention
    """

    RECONSTRUCTION_ALGORITHM_DMAS = "DMAS"
    """
    Corresponds to the reconstruction algorithm DMAS with the MitkBeamformingAdapter.\n
    Usage: module image_reconstruction, naming convention
    """

    RECONSTRUCTION_ALGORITHM_SDMAS = "sDMAS"
    """
    Corresponds to the reconstruction algorithm sDMAS with the MitkBeamformingAdapter.\n
    Usage: module image_reconstruction, naming convention
    """

    RECONSTRUCTION_ALGORITHM_PYTORCH_DAS = "PyTorch_DAS"
    """
    Corresponds to the reconstruction algorithm DAS with the PyTorchDASAdapter.\n
    Usage: module image_reconstruction, naming convention
    """

    RECONSTRUCTION_ALGORITHM_TIME_REVERSAL = "time_reversal"
    """
    Corresponds to the reconstruction algorithm Time Reversal with TimeReversalAdapter.\n
    Usage: module image_reconstruction, naming convention
    """

    RECONSTRUCTION_ALGORITHM_TEST = "TEST"
    """
    Corresponds to an adapter for testing purposes only.\n
    Usage: module image_reconstruction, naming convention
    """

    RECONSTRUCTION_INVERSE_CRIME = ("reconstruction_inverse_crime", (bool, np.bool, np.bool_))
    """
    If True, the Time Reversal reconstruction will commit the "inverse crime".\n
    Usage: TimeReversalAdapter
    """

    RECONSTRUCTION_MITK_BINARY_PATH = ("reconstruction_mitk_binary_path", str)
    """
    Absolute path to the Mitk Beamforming script.\n
    Usage: adapter MitkBeamformingAdapter
    """

    RECONSTRUCTION_MITK_SETTINGS_XML = ("reconstruction_mitk_settings_xml", str)
    """
    Absolute path to the Mitk Beamforming script settings.\n
    Usage: adapter MitkBeamformingAdapter
    """

    RECONSTRUCTION_BMODE_METHOD = ("reconstruction_bmode_method", str)
    """
    Choice of the B-Mode method used in the Mitk Beamforming.\n
    Usage: adapter MitkBeamformingAdapter
    """

    RECONSTRUCTION_BMODE_METHOD_ABS = "Abs"
    """
    Corresponds to the absolute value as the B-Mode method used in the Mitk Beamforming.\n
    Usage: adapter MitkBeamformingAdapter, naming convention
    """

    RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM = "EnvelopeDetection"
    """
    Corresponds to the Hilbert transform as the B-Mode method used in the Mitk Beamforming.\n
    Usage: adapter MitkBeamformingAdapter, naming convention
    """

    RECONSTRUCTION_BMODE_BEFORE_RECONSTRUCTION = "Envelope_Detection_before_Reconstruction"
    """
    Specifies whether an envelope detection should be performed before reconstruction, default is False
    Usage: adapter PyTorchDASAdapter, naming convention
    """

    RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION = "Envelope_Detection_after_Reconstruction"
    """
    Specifies whether an envelope detection should be performed after reconstruction, default is False
    Usage: adapter PyTorchDASAdapter, naming convention
    """

    RECONSTRUCTION_APODIZATION_METHOD = ("reconstruction_apodization_method", str)
    """
    Choice of the apodization method used, i.e. window functions .\n
    Usage: adapter PyTorchDASAdapter
    """

    RECONSTRUCTION_APODIZATION_BOX = "BoxApodization"
    """
    Corresponds to the box window function for apodization.\n
    Usage: adapter PyTorchDASAdapter, naming convention
    """

    RECONSTRUCTION_APODIZATION_HANN = "HannApodization"
    """
    Corresponds to the Hann window function for apodization.\n
    Usage: adapter PyTorchDASAdapter, naming convention
    """

    RECONSTRUCTION_APODIZATION_HAMMING = "HammingApodization"
    """
    Corresponds to the Hamming window function for apodization.\n
    Usage: adapter PyTorchDASAdapter, naming convention
    """

    RECONSTRUCTION_PERFORM_BANDPASS_FILTERING = ("reconstruction_perform_bandpass_filtering",
                                    (bool, np.bool, np.bool_))
    """
    Whether bandpass filtering should be applied or not. Default should be True\n
    Usage: adapter PyTorchDASAdapter
    """

    TUKEY_WINDOW_ALPHA = ("tukey_window_alpha", (int, np.integer, float, np.float))
    """
    Sets alpha value of Tukey window between 0 (similar to box window) and 1 (similar to Hann window).
    Default is 0.5\n
    Usage: adapter PyTorchDASAdapter
    """

    BANDPASS_CUTOFF_LOWPASS = ("bandpass_cuttoff_lowpass", (int, np.integer, float, np.float))
    """
    Sets the cutoff threshold in MHz for lowpass filtering, i.e. upper limit of the tukey filter. Default is 8 MHz\n
    Usage: adapter PyTorchDASAdapter
    """

    BANDPASS_CUTOFF_HIGHPASS = ("bandpass_cuttoff_highpass", (int, np.integer, float, np.float))
    """
    Sets the cutoff threshold in MHz for highpass filtering, i.e. lower limit of the tukey filter. Default is 0.1 MHz\n
    Usage: adapter PyTorchDASAdapter
    """

    RECONSTRUCTED_DATA = "reconstructed_data"
    """
    Name of the reconstructed data field in the SIMPA output file.\n
    Usage: naming convention
    """

    RECONSTRUCTED_DATA_NOISE = "reconstructed_data_noise"
    """
    Name of the reconstructed data with applied noise field in the SIMPA output file.\n
    Usage: naming convention
    """

    RECONSTRUCTION_MODE = ("reconstruction_mode", str)
    """
    Choice of the reconstruction mode used in the Backprojection.\n
    Usage: adapter BackprojectionAdapter
    """

    RECONSTRUCTION_MODE_DIFFERENTIAL = "differential"
    """
    Corresponds to the differential mode used in the Backprojection.\n
    Usage: adapter BackprojectionAdapter, naming_convention
    """

    RECONSTRUCTION_MODE_PRESSURE = "pressure"
    """
    Corresponds to the pressure mode used in the Backprojection.\n
    Usage: adapter BackprojectionAdapter, naming_convention
    """

    RECONSTRUCTION_MODE_FULL = "full"
    """
    Corresponds to the full mode used in the Backprojection.\n
    Usage: adapter BackprojectionAdapter, naming_convention
    """

    """
    Upsampling settings
    """

    CROP_IMAGE = ("crop_image", bool)
    """
    If True, the PA image cropped in the image processing.\n
    Usage: module process
    """

    CROP_POWER_OF_TWO = ("crop_power_of_two", bool)
    """
    If True, the PA image cropped to the shape as the nearest power of two in the image processing.\n
    Usage: module process
    """

    PERFORM_UPSAMPLING = ("sample", bool)
    """
    If True, the PA image upsampled in the image processing.\n
    Usage: module process
    """

    UPSAMPLING_METHOD = ("upsampling_method", str)
    """
    Choice of the upsampling method used in the image processing.\n
    Usage: module process
    """

    UPSAMPLING_METHOD_DEEP_LEARNING = "deeplearning"
    """
    Corresponds to deep learning as the upsampling method used in the image processing.\n
    Usage: module process, naming concention
    """

    UPSAMPLING_METHOD_NEAREST_NEIGHBOUR = "nearestneighbour"
    """
    Corresponds to nearest neighbour as the upsampling method used in the image processing.\n
    Usage: module process, naming concention
    """

    UPSAMPLING_METHOD_BILINEAR = "bilinear"
    """
    Corresponds to the bilinear upsampling method used in the image processing.\n
    Usage: module process, naming concention
    """

    UPSAMPLING_METHOD_LANCZOS2 = "lanczos2"
    """
    Corresponds to lanczos with kernel size 2 as the upsampling method used in the image processing.\n
    Usage: module process, naming concention
    """

    UPSAMPLING_METHOD_LANCZOS3 = "lanczos3"
    """
    Corresponds to lanczos with kernel size 3 as the upsampling method used in the image processing.\n
    Usage: module process, naming concention
    """

    UPSAMPLING_SCRIPT = ("upsampling_script", str)
    """
    Name of the upsampling script used for the lanczos upsampling.\n
    Usage: module process
    """

    UPSAMPLING_SCRIPT_LOCATION = ("upsampling_script_location", str)
    """
    Absolute path to the upsampling script used for the lanczos upsampling.\n
    Usage: module process
    """

    UPSCALE_FACTOR = ("upscale_factor", (int, float, np.int_, np.float_))
    """
    Upscale factor of the upsampling in the image processing.\n
    Usage: module process
    """

    DL_MODEL_PATH = ("dl_model_path", str)
    """
    Absolute path to the deep learning model used for the deep learning upsampling.\n
    Usage: module process
    """

    # physical property volume types
    PROPERTY_ABSORPTION_PER_CM = "mua"
    """
    Optical absorption of the generated volume/structure in 1/cm.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_SCATTERING_PER_CM = "mus"
    """
    Optical scattering (NOT REDUCED SCATTERING mus'! mus'=mus*(1-g) ) of the generated volume/structure in 1/cm.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_ANISOTROPY = "g"
    """
    Optical scattering anisotropy of the generated volume/structure.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_OXYGENATION = "oxy"
    """
    Oxygenation of the generated volume/structure.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_SEGMENTATION = "seg"
    """
    Segmentation of the generated volume/structure.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_GRUNEISEN_PARAMETER = "gamma"
    """
    We define PROPERTY_GRUNEISEN_PARAMETER to contain all wavelength-independent constituents of the PA signal.
    This means that it contains the percentage of absorbed light converted into heat.
    Naturally, one could make an argument that this should not be the case, however, it simplifies the usage of 
    this tool.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_SPEED_OF_SOUND = "sos"
    """
    Speed of sound of the generated volume/structure in m/s.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_DENSITY = "density"
    """
    Density of the generated volume/structure in kg/m³.\n
    Usage: SIMPA package, naming convention
    """

    PROPERTY_ALPHA_COEFF = "alpha_coeff"
    """
    Acoustic attenuation of kwave of the generated volume/structure in dB/cm/MHz.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    PROPERTY_SENSOR_MASK = "sensor_mask"
    """
    Sensor mask of kwave of the used PA device.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    PROPERTY_DIRECTIVITY_ANGLE = "directivity_angle"
    """
    Directionality of the sensors in kwave of the used PA device.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    PROPERTY_ALPHA_POWER = ("medium_alpha_power", (int, np.integer, float, np.float))
    """
    Exponent of the exponential acoustic attenuation law of kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    # Volume geometry settings
    SPACING_MM = ("voxel_spacing_mm", (int, np.integer, float, np.float))
    """
    Isotropic extent of one voxels in mm in the generated volume.\n
    Usage: SIMPA package
    """

    DIM_VOLUME_X_MM = ("volume_x_dim_mm", (int, np.integer, float, np.float))
    """
    Extent of the x-axis of the generated volume.\n
    Usage: SIMPA package
    """

    DIM_VOLUME_Y_MM = ("volume_y_dim_mm", (int, np.integer, float, np.float))
    """
    Extent of the y-axis of the generated volume.\n
    Usage: SIMPA package
    """

    DIM_VOLUME_Z_MM = ("volume_z_dim_mm", (int, np.integer, float, np.float))
    """
    Extent of the z-axis of the generated volume.\n
    Usage: SIMPA package
    """

    # PML parameters
    PMLSize = ("pml_size", (list, tuple, np.ndarray))
    """
    Size of the "perfectly matched layer" (PML) around the simulated volume in kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    PMLAlpha = ("pml_alpha", (int, np.integer, float, np.float))
    """
    Alpha coefficient of the "perfectly matched layer" (PML) around the simulated volume in kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    PMLInside = ("pml_inside", (bool, np.bool, np.bool_))
    """
    If True, the "perfectly matched layer" (PML) in kwave is located inside the volume.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    PlotPML = ("plot_pml", (bool, np.bool, np.bool_))
    """
    If True, the "perfectly matched layer" (PML) around the simulated volume in kwave is plotted.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    # Acoustic Sensor Properties
    SENSOR_RECORD = ("sensor_record", str)
    """
    Sensor Record mode of the sensor in kwave. Default should be "p".\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    SENSOR_CENTER_FREQUENCY_HZ = ("sensor_center_frequency", (int, np.integer, float, np.float))
    """
    Sensor center frequency in kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    SENSOR_BANDWIDTH_PERCENT = ("sensor_bandwidth", (int, np.integer, float, np.float))
    """
    Sensor bandwidth in kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    SENSOR_DIRECTIVITY_SIZE_M = ("sensor_directivity_size", (int, np.integer, float, np.float))
    """
    Size of each detector element in kwave.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    SENSOR_DIRECTIVITY_PATTERN = "sensor_directivity_pattern"
    """
    Sensor directivity pattern of the sensor in kwave. Default should be "pressure".\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    SENSOR_SAMPLING_RATE_MHZ = ("sensor_sampling_rate_mhz", (int, np.integer, float, np.float))
    """
    Sampling rate of the used PA device.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    SENSOR_NUM_ELEMENTS = ("sensor_num_elements", (int, np.integer))
    """
    Number of detector elements for kwave if no device was selected.\n
    Usage: adapter KwaveAcousticForwardModel, adapter TimeReversalAdapter, naming convention
    """

    SENSOR_NUM_USED_ELEMENTS = ("sensor_num_used_elements", (int, np.integer))
    """
    Number of detector elements that fit into the generated volume if the dimensions and/or spacing of the generated 
    volume were not highly resolved enough to be sufficient for the selected PA device.\n
    Usage: module acoustic_simulation, naming convention
    """

    SENSOR_CONCAVE = "concave"
    """
    Indicates that the geometry of the used PA device in the Mitk Beamforming is concave.\n
    Usage: adapter MitkBeamformingAdapter, naming convention
    """

    SENSOR_LINEAR = "linear"
    """
    Indicates that the geometry of the used PA device in the Mitk Beamforming is linear.\n
    Usage: adapter MitkBeamformingAdapter, naming convention
    """

    SENSOR_RADIUS_MM = "sensor_radius_mm"
    """
    Radius of a concave geometry of the used PA device in the Mitk Beamforming.\n
    Usage: adapter MitkBeamformingAdapter, naming convention
    """

    # Noise properties
    APPLY_NOISE_MODEL = ("apply_noise_model", bool)
    """
    If True, the simulation will apply a noise model.\n
    Usage: module core (simulate.py)
    """

    NOISE_MODEL = ("noise_model", str)
    """
    Choice of the noise model.\n 
    Usage: module noise_simulation
    """

    NOISE_MODEL_GAUSSIAN = "noise_model_gaussian"
    """
    Corresponds to a gaussian noise model.\n 
    Usage: module noise_simulation
    """

    NOISE_MEAN = ("noise_mean", (int, np.integer, float, np.float))
    """
    Mean of the gaussian noise model used in the noise modelling.\n 
    Usage: module noise_simulation
    """

    NOISE_STD = ("noise_std", (int, np.integer, float, np.float))
    """
    Standard deviation of the gaussian noise model used in the noise modelling.\n 
    Usage: module noise_simulation
    """

    NOISE_MODEL_PATH = ("noise_model_path", str)
    """
    Absolute path of a .csv file with an experimentally recorded noise model.\n
    Usage: module noise_simulation
    """

    # Structures
    STRUCTURES = ("structures", dict)
    """
    Settings dictionary which contains all the structures that should be generated inside the volume.\n
    Usage: module volume_creation
    """

    HORIZONTAL_LAYER_STRUCTURE = "HorizontalLayerStructure"
    """
    Corresponds to the HorizontalLayerStructure in the structure_library.\n
    Usage: module volume_creation, naming_convention
    """

    CIRCULAR_TUBULAR_STRUCTURE = "CircularTubularStructure"
    """
    Corresponds to the CircularTubularStructure in the structure_library.\n
    Usage: module volume_creation, naming_convention
    """

    ELLIPTICAL_TUBULAR_STRUCTURE = "EllipticalTubularStructure"
    """
    Corresponds to the EllipticalTubularStructure in the structure_library.\n
    Usage: module volume_creation, naming_convention
    """

    SPHERICAL_STRUCTURE = "SphericalStructure"
    """
    Corresponds to the SphericalStructure in the structure_library.\n
    Usage: module volume_creation, naming_convention
    """

    PARALLELEPIPED_STRUCTURE = "ParallelepipedStructure"
    """
    Corresponds to the ParallelepipedStructure in the structure_library.\n
    Usage: module volume_creation, naming_convention
    """

    RECTANGULAR_CUBOID_STRUCTURE = "RectangularCuboidStructure"
    """
    Corresponds to the RectangularCuboidStructure in the structure_library.\n
    Usage: module volume_creation, naming_convention
    """

    STRUCTURE_TYPE = ("structure_type", str)
    """
    Defines the structure type to one structure in the structure_library.\n
    Usage: module volume_creation
    """

    STRUCTURE_SEGMENTATION_TYPE = "structure_segmentation_type"
    """
    Defines the structure segmentation type to one segmentation type in SegmentationClasses.\n
    Usage: module volume_creation, naming convention
    """

    UNITS_ARBITRARY = "arbitrary_unity"
    """
    Define arbitrary units if no units were given in the settings.\n
    Usage: module optical_simulation, naming convention
    """

    UNITS_PRESSURE = "newton_per_meters_squared"
    """
    Standard units used in the SIMPA framework.\n
    Usage: module optical_simulation, naming convention
    """

    """
    IO settings
    """

    SIMPA_OUTPUT_PATH = ("simpa_output_path", str)
    """
    Default path of the SIMPA output if not specified otherwise.\n
    Usage: SIMPA package
    """

    SIMPA_OUTPUT_NAME = "simpa_output.hdf5"
    """
    Default filename of the SIMPA output if not specified otherwise.\n
    Usage: SIMPA package, naming convention
    """

    SETTINGS_JSON = ("settings_json", (bool, np.bool_))
    """
    If True, the SIMPA settings are saved in a .json file.\n
    Usage: SIMPA package
    """

    SETTINGS_JSON_PATH = ("settings_json_path", str)
    """
    Absolute path to a .json file if SETTINGS_JSON is set to True.
    Usage: SIMPA package
    """

    SETTINGS = "settings"
    """
    Location of the simulation settings in the SIMPA output file.\n
    Usage: naming convention
    """

    SIMULATION_PROPERTIES = "simulation_properties"
    """
    Location of the simulation properties in the SIMPA output file.\n
    Usage: naming convention
    """

    SIMULATIONS = "simulations"
    """
    Location of the simulation outputs in the SIMPA output file.\n
    Usage: naming convention
    """

    UPSAMPLED_DATA = "upsampled_data"
    """
    Name of the simulation outputs as upsampled data in the SIMPA output file.\n
    Usage: naming convention
    """

    ORIGINAL_DATA = "original_data"
    """
    Name of the simulation outputs as original data in the SIMPA output file.\n
    Usage: naming convention
    """
