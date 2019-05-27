class Tags:

    # General settings
    SIMULATION_PATH = "simulation_path"
    VOLUME_NAME = "volume_name"
    WAVELENGTHS = "wavelengths"
    WAVELENGTH = "wavelength"
    RANDOM_SEED = "random_seed"
    TISSUE_PROPERTIES_OUPUT_NAME = "properties"

    # Optical model settings
    RUN_OPTICAL_MODEL = 'run_optical_forward_model'
    OPTICAL_MODEL_OUTPUT_NAME = "optical_forward_model_output"
    OPTICAL_MODEL_BINARY_PATH = "optical_model_binary_path"
    OPTICAL_MODEL_NUMBER_PHOTONS = "optical_model_number_of_photons"
    OPTICAL_MODEL_PROBE_XML_FILE = "optical_model_probe_xml_file"

    # Acoustic model settings
    RUN_ACOUSTIC_MODEL = 'run_acoustic_forward_model'
    ACOUSTIC_MODEL_OUTPUT_NAME = "acoustic_forward_model_output"

    # physical property volume types
    PROPERTY_ABSORPTION = 'mua'
    PROPERTY_SCATTERING = 'mus'
    PROPERTY_ANISOTROPY = 'g'
    PROPERTY_OXYGENATION = 'sO2'
    PROPERTY_SEGMENTATION = 'segmentation'

    # Air layer
    AIR_LAYER_HEIGHT_MM = "air_layer_height"

    # Gel Pad Layer
    GELPAD_LAYER_HEIGHT_MM = "gelpad_layer_height_mm"

    # Volume geometry settings
    SPACING_MM = "voxel_spacing_mm"
    DIM_VOLUME_X_MM = "volume_x_dim_mm"
    DIM_VOLUME_Y_MM = "volume_y_dim_mm"
    DIM_VOLUME_Z_MM = "volume_z_dim_mm"

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
    STRUCTURES = "structures"
    CHILD_STRUCTURES = "child_structures"
    STRUCTURE_TYPE = "structure_type"
    STRUCTURE_SEGMENTATION_TYPE = "structure_segmentation_type"
    STRUCTURE_TISSUE_PROPERTIES = "structure_tissue_properties"

    STRUCTURE_DEPTH_MIN_MM = "structure_depth_min_mm"
    STRUCTURE_DEPTH_MAX_MM = "structure_depth_max_mm"

    STRUCTURE_USE_DISTORTION = "structure_distortion_multiplicative"
    STRUCTURE_DISTORTED_PARAM_LIST = "structure_distorted_param_list"
    STRUCTURE_DISTORTION_WAVELENGTH_MM = "structure_distortion_wavelength_mm"

    STRUCTURE_BACKGROUND = "structure_background"

    STRUCTURE_LAYER = "structure_layer"
    STRUCTURE_THICKNESS_MIN_MM = "structure_thickness_min_mm"
    STRUCTURE_THICKNESS_MAX_MM = "structure_thickness_max_mm"

    STRUCTURE_TUBE = "structure_tube"
    STRUCTURE_RADIUS_MIN_MM = "structure_radius_min_mm"
    STRUCTURE_RADIUS_MAX_MM = "structure_radius_max_mm"
    STRUCTURE_FORCE_ORTHAGONAL_TO_PLANE = "structure_force_orthagonal_to_plane"
    STRUCTURE_TUBE_START_X_MIN_MM = "structure_tube_start_x_min_mm"
    STRUCTURE_TUBE_START_X_MAX_MM = "structure_tube_start_x_max_mm"


class SegmentationClasses:
    AIR = 0
    MUSCLE = 1
    BONE = 2
    BLOOD = 3
    EPIDERMIS = 4
    DERMIS = 5
    FAT = 6
    ULTRASOUND_GEL_PAD = 7

class StandardProperties:
    AIR_MUA = 1e-7
    AIR_MUS = 1e-7
    AIR_G = 1