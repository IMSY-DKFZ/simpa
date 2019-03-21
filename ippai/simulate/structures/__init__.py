from ippai.simulate import Tags, SegmentationClasses
from ippai.simulate.tissue_properties import get_epidermis_settings, get_dermis_settings, \
    get_subcutaneous_fat_settings, get_blood_settings, \
    get_arterial_blood_settings, get_venous_blood_settings, get_bone_settings, get_muscle_settings


def create_vessel_tube(x_min=None, x_max=None, z_min=None, z_max=None, r_min=0.5, r_max=3.0):
    vessel_dict = dict()
    vessel_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    vessel_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = z_min
    vessel_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = z_max
    vessel_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = r_min
    vessel_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = r_max
    vessel_dict[Tags.STRUCTURE_TUBE_START_X_MIN_MM] = x_min
    vessel_dict[Tags.STRUCTURE_TUBE_START_X_MAX_MM] = x_max
    vessel_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_blood_settings()
    vessel_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
    return vessel_dict


# #############################################
# FOREARM MODEL STRUCTURES
# #############################################
def create_forearm_structures(relative_shift_mm=0, background_oxy=0.0):

    structures_dict = dict()
    structures_dict["muscle"] = create_muscle_background(background_oxy=background_oxy)
    structures_dict["subcutaneous_fat"] = create_subcutaneous_fat_layer(background_oxy=background_oxy)
    structures_dict["dermis"] = create_dermis_layer(background_oxy=background_oxy)
    structures_dict["epidermis"] = create_epidermis_layer(background_oxy=background_oxy)
    structures_dict["radial_artery"] = create_radial_artery(relative_shift_mm)
    structures_dict["ulnar_artery"] = create_ulnar_artery(relative_shift_mm)
    structures_dict["interosseous_artery"] = create_interosseous_artery(relative_shift_mm)
    structures_dict["radius"] = create_radius_bone(relative_shift_mm)
    structures_dict["ulna"] = create_ulna_bone(relative_shift_mm)

    for i in range(7):
        position = relative_shift_mm - 17.5 + i * 10
        structures_dict["subcutaneous_vessel_"+str(i+1)] = create_subcutaneous_vein(position)
    return structures_dict


def create_muscle_background(background_oxy=0.0):
    muscle_dict = dict()
    muscle_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    muscle_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_muscle_settings(background_oxy=background_oxy)
    muscle_dict[Tags.STRUCTURE_USE_DISTORTION] = False
    muscle_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_B, Tags.KEY_OXY, Tags.KEY_W]
    muscle_dict[Tags.STRUCTURE_DISTORTION_WAVELENGTH_MM] = 2
    muscle_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.MUSCLE
    return muscle_dict


def create_epidermis_layer(background_oxy=0.0):
    epidermis_dict = dict()
    epidermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    epidermis_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 0
    epidermis_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 0
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = 0.06
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = 0.06
    epidermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_epidermis_settings(background_oxy=background_oxy)
    epidermis_dict[Tags.STRUCTURE_USE_DISTORTION] = False
    epidermis_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_OXY, Tags.KEY_W]
    epidermis_dict[Tags.STRUCTURE_DISTORTION_WAVELENGTH_MM] = 2
    epidermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.EPIDERMIS
    return epidermis_dict


def create_dermis_layer(background_oxy=0.0):
    dermis_dict = dict()
    dermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    dermis_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 0
    dermis_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 0
    dermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = 1.8
    dermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = 2.2
    dermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_dermis_settings(background_oxy=background_oxy)
    dermis_dict[Tags.STRUCTURE_USE_DISTORTION] = False
    dermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.DERMIS
    dermis_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_OXY, Tags.KEY_W]
    dermis_dict[Tags.STRUCTURE_DISTORTION_WAVELENGTH_MM] = 2
    return dermis_dict


def create_subcutaneous_fat_layer(background_oxy=0.0):
    fat_dict = dict()
    fat_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    fat_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 1.5
    fat_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 1.5
    fat_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = 1.5
    fat_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = 1.9
    fat_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_subcutaneous_fat_settings(background_oxy=background_oxy)
    fat_dict[Tags.STRUCTURE_USE_DISTORTION] = False
    fat_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.FAT
    fat_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_F, Tags.KEY_OXY, Tags.KEY_W]
    fat_dict[Tags.STRUCTURE_DISTORTION_WAVELENGTH_MM] = 2
    return fat_dict


def create_radial_artery(relative_shift_mm=0.0):
    radial_dict = create_vessel_tube(x_min=-1+relative_shift_mm, x_max=6+relative_shift_mm,
                                     z_min=7, z_max=9, r_min=1, r_max=1.4)
    radial_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_arterial_blood_settings()
    radial_dict[Tags.CHILD_STRUCTURES] = dict()
    radial_dict[Tags.CHILD_STRUCTURES]["left_radial_accompanying_vein"] = create_vessel_tube(x_min=-2.7, x_max=-2.3,
                                                                                      z_min=-0.75, z_max=0.75,
                                                                                      r_min=0.4, r_max=0.6)
    radial_dict[Tags.CHILD_STRUCTURES]["left_radial_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    radial_dict[Tags.CHILD_STRUCTURES]["right_radial_accompanying_vein"] = create_vessel_tube(x_min=2.3, x_max=2.7,
                                                                                       z_min=-0.75, z_max=0.75,
                                                                                       r_min=0.4, r_max=0.6)
    radial_dict[Tags.CHILD_STRUCTURES]["right_radial_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    return radial_dict


def create_ulnar_artery(relative_shift_mm=0.0):
    ulnar_dict = create_vessel_tube(x_min=29 + relative_shift_mm, x_max=36 + relative_shift_mm,
                                    z_min=7, z_max=9, r_min=1, r_max=1.3)
    ulnar_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_arterial_blood_settings()
    ulnar_dict[Tags.CHILD_STRUCTURES] = dict()
    ulnar_dict[Tags.CHILD_STRUCTURES]["left_ulnar_accompanying_vein"] = create_vessel_tube(x_min=-2.7, x_max=-2.3,
                                                                                      z_min=-0.75, z_max=0.75,
                                                                                      r_min=0.4, r_max=0.6)
    ulnar_dict[Tags.CHILD_STRUCTURES]["left_ulnar_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    ulnar_dict[Tags.CHILD_STRUCTURES]["right_ulnar_accompanying_vein"] = create_vessel_tube(x_min=2.3, x_max=2.7,
                                                                                       z_min=-0.75, z_max=0.75,
                                                                                       r_min=0.4, r_max=0.6)
    ulnar_dict[Tags.CHILD_STRUCTURES]["right_ulnar_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    return ulnar_dict


def create_interosseous_artery(relative_shift_mm=0.0):
    inter_dict = create_vessel_tube(x_min=15 + relative_shift_mm, x_max=20 + relative_shift_mm,
                                           z_min=18, z_max=20, r_min=0.4, r_max=0.6)
    inter_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_arterial_blood_settings()
    inter_dict[Tags.CHILD_STRUCTURES] = dict()
    inter_dict[Tags.CHILD_STRUCTURES]["left_inter_accompanying_vein"] = create_vessel_tube(x_min=-1.7, x_max=-1.3,
                                                                                      z_min=-0.5, z_max=0.5,
                                                                                      r_min=0.2, r_max=0.3)
    inter_dict[Tags.CHILD_STRUCTURES]["left_inter_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    inter_dict[Tags.CHILD_STRUCTURES]["right_inter_accompanying_vein"] = create_vessel_tube(x_min=1.3, x_max=1.7,
                                                                                       z_min=-0.5, z_max=0.5,
                                                                                       r_min=0.2, r_max=0.3)
    inter_dict[Tags.CHILD_STRUCTURES]["right_inter_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    return inter_dict


def create_radius_bone(relative_shift_mm=0.0):
    radius_dict = create_vessel_tube(x_min=-1 + relative_shift_mm, x_max=1 + relative_shift_mm,
                                     z_min=20, z_max=23, r_min=8, r_max=10)
    radius_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_bone_settings()
    radius_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BONE
    return radius_dict


def create_ulna_bone(relative_shift_mm=0.0):
    radius_dict = create_vessel_tube(x_min=34 + relative_shift_mm, x_max=35 + relative_shift_mm,
                                     z_min=20, z_max=23, r_min=6, r_max=8)
    radius_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_bone_settings()
    radius_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BONE
    return radius_dict


def create_subcutaneous_vein(relative_shift_mm=0.0):
    interosseous_dict = create_vessel_tube(x_min=-5 + relative_shift_mm, x_max=5 + relative_shift_mm,
                                           z_min=0.8, z_max=2.3, r_min=0.1, r_max=1)
    interosseous_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_venous_blood_settings()
    return interosseous_dict
