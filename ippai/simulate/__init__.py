class Tags:

    # General settings
    SIMULATION_PATH = "simulation_path"
    WAVELENGTH = "wavelength"
    RANDOM_SEED = "random_seed"

    # Air layer
    AIR_LAYER_HEIGHT_MM = "air_layer_height"

    # Model settings
    RUN_OPTICAL_MODEL = 'run_optical_forward_model'
    RUN_ACOUSTIC_MODEL = 'run_acoustic_forward_model'

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
