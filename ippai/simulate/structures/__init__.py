from ippai.simulate import Tags, SegmentationClasses, TissueProperties
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
    muscle_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = 2
    muscle_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.MUSCLE
    return muscle_dict


def create_epidermis_layer(background_oxy=0.0):
    epidermis_dict = dict()
    epidermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    epidermis_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 0
    epidermis_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 0
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = TissueProperties.EPIDERMIS_THICKNESS_MEAN_MM - TissueProperties.EPIDERMIS_THICKNESS_STD_MM
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = TissueProperties.EPIDERMIS_THICKNESS_MEAN_MM + TissueProperties.EPIDERMIS_THICKNESS_STD_MM
    epidermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_epidermis_settings(background_oxy=background_oxy)
    epidermis_dict[Tags.STRUCTURE_USE_DISTORTION] = False
    epidermis_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_M, Tags.KEY_OXY, Tags.KEY_W]
    epidermis_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = 2
    epidermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.EPIDERMIS
    return epidermis_dict


def create_dermis_layer(background_oxy=0.0):
    dermis_dict = dict()
    dermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    dermis_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 0
    dermis_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 0
    dermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = TissueProperties.DERMIS_THICKNESS_MEAN_MM - TissueProperties.DERMIS_THICKNESS_STD_MM
    dermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = TissueProperties.DERMIS_THICKNESS_MEAN_MM + TissueProperties.DERMIS_THICKNESS_STD_MM
    dermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_dermis_settings(background_oxy=background_oxy)
    dermis_dict[Tags.STRUCTURE_USE_DISTORTION] = False
    dermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.DERMIS
    dermis_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_OXY, Tags.KEY_W]
    dermis_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = 2
    return dermis_dict


def create_subcutaneous_fat_layer(background_oxy=0.0):
    """
    FIXME: DEPRECATED
    FIXME: first research into fat tissue properties before adding this to simulation (!)
    :param background_oxy:
    :return:
    """
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
    fat_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = 2
    return fat_dict


def create_radial_artery(relative_shift_mm=0.0):
    radial_dict = create_vessel_tube(x_min=relative_shift_mm + TissueProperties.RADIAL_ARTERY_X_POSITION_MEAN_MM - TissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                     x_max=relative_shift_mm + TissueProperties.RADIAL_ARTERY_X_POSITION_MEAN_MM + TissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                     z_min=TissueProperties.RADIAL_ARTERY_DEPTH_MEAN_MM - TissueProperties.RADIAL_ARTERY_DEPTH_STD_MM,
                                     z_max=TissueProperties.RADIAL_ARTERY_DEPTH_MEAN_MM + TissueProperties.RADIAL_ARTERY_DEPTH_STD_MM,
                                     r_min=TissueProperties.RADIAL_ARTERY_DIAMETER_MEAN_MM / 2 - TissueProperties.RADIAL_ARTERY_DIAMETER_STD_MM / 2,
                                     r_max=TissueProperties.RADIAL_ARTERY_DIAMETER_MEAN_MM / 2 + TissueProperties.RADIAL_ARTERY_DIAMETER_STD_MM / 2)
    radial_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_arterial_blood_settings()
    radial_dict[Tags.CHILD_STRUCTURES] = dict()
    radial_dict[Tags.CHILD_STRUCTURES]["left_radial_accompanying_vein"] = create_vessel_tube(x_min=- TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                             x_max=- TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                             z_min=- TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                             z_max=TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                             r_min=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2,
                                                                                             r_max=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2)
    radial_dict[Tags.CHILD_STRUCTURES]["left_radial_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    radial_dict[Tags.CHILD_STRUCTURES]["right_radial_accompanying_vein"] = create_vessel_tube(x_min=TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                              x_max=TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                              z_min=- TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                              z_max=TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                              r_min=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2,
                                                                                              r_max=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2)
    radial_dict[Tags.CHILD_STRUCTURES]["right_radial_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    return radial_dict


def create_ulnar_artery(relative_shift_mm=0.0):
    ulnar_dict = create_vessel_tube(x_min=relative_shift_mm + TissueProperties.ULNAR_ARTERY_X_POSITION_MEAN_MM - TissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    x_max=relative_shift_mm + TissueProperties.ULNAR_ARTERY_X_POSITION_MEAN_MM + TissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    z_min=TissueProperties.ULNAR_ARTERY_DEPTH_MEAN_MM - TissueProperties.ULNAR_ARTERY_DEPTH_STD_MM,
                                    z_max=TissueProperties.ULNAR_ARTERY_DEPTH_MEAN_MM + TissueProperties.ULNAR_ARTERY_DEPTH_STD_MM,
                                    r_min=TissueProperties.ULNAR_ARTERY_DIAMETER_MEAN_MM / 2 - TissueProperties.ULNAR_ARTERY_DIAMETER_STD_MM / 2,
                                    r_max=TissueProperties.ULNAR_ARTERY_DIAMETER_MEAN_MM / 2 + TissueProperties.ULNAR_ARTERY_DIAMETER_STD_MM / 2)
    ulnar_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_arterial_blood_settings()
    ulnar_dict[Tags.CHILD_STRUCTURES] = dict()
    ulnar_dict[Tags.CHILD_STRUCTURES]["left_ulnar_accompanying_vein"] = create_vessel_tube(x_min=- TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                           x_max=- TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                           z_min=- TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                           z_max=TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                           r_min=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2,
                                                                                           r_max=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2)
    ulnar_dict[Tags.CHILD_STRUCTURES]["left_ulnar_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    ulnar_dict[Tags.CHILD_STRUCTURES]["right_ulnar_accompanying_vein"] = create_vessel_tube(x_min=TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                            x_max=TissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + TissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM,
                                                                                            z_min=- TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                            z_max=TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                            r_min=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2,
                                                                                            r_max=TissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + TissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2)
    ulnar_dict[Tags.CHILD_STRUCTURES]["right_ulnar_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    return ulnar_dict


def create_interosseous_artery(relative_shift_mm=0.0):
    inter_dict = create_vessel_tube(x_min=relative_shift_mm + TissueProperties.MEDIAN_ARTERY_X_POSITION_MEAN_MM - TissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    x_max=relative_shift_mm + TissueProperties.MEDIAN_ARTERY_X_POSITION_MEAN_MM + TissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    z_min=TissueProperties.MEDIAN_ARTERY_DEPTH_MEAN_MM - TissueProperties.MEDIAN_ARTERY_DEPTH_STD_MM,
                                    z_max=TissueProperties.MEDIAN_ARTERY_DEPTH_MEAN_MM + TissueProperties.MEDIAN_ARTERY_DEPTH_STD_MM,
                                    r_min=TissueProperties.MEDIAN_ARTERY_DIAMETER_MEAN_MM / 2 - TissueProperties.MEDIAN_ARTERY_DIAMETER_STD_MM / 2,
                                    r_max=TissueProperties.MEDIAN_ARTERY_DIAMETER_MEAN_MM / 2 + TissueProperties.MEDIAN_ARTERY_DIAMETER_STD_MM / 2)
    inter_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_arterial_blood_settings()
    inter_dict[Tags.CHILD_STRUCTURES] = dict()
    inter_dict[Tags.CHILD_STRUCTURES]["left_inter_accompanying_vein"] = create_vessel_tube(x_min=- TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM - TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM,
                                                                                           x_max=- TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM + TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM,
                                                                                           z_min=- TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                           z_max=TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                           r_min=TissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 - TissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2,
                                                                                           r_max=TissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 + TissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2)
    inter_dict[Tags.CHILD_STRUCTURES]["left_inter_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    inter_dict[Tags.CHILD_STRUCTURES]["right_inter_accompanying_vein"] = create_vessel_tube(x_min=TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM - TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM,
                                                                                            x_max=TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM + TissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM,
                                                                                            z_min=- TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                            z_max=TissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2,
                                                                                            r_min=TissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 - TissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2,
                                                                                            r_max=TissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 + TissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2)
    inter_dict[Tags.CHILD_STRUCTURES]["right_inter_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        get_venous_blood_settings()
    return inter_dict


def create_radius_bone(relative_shift_mm=0.0):
    radius_dict = create_vessel_tube(x_min=relative_shift_mm - TissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM,
                                     x_max=relative_shift_mm + TissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM,
                                     z_min=TissueProperties.RADIUS_BONE_DEPTH_MEAN_MM - TissueProperties.RADIUS_BONE_DEPTH_STD_MM,
                                     z_max=TissueProperties.RADIUS_BONE_DEPTH_MEAN_MM + TissueProperties.RADIUS_BONE_DEPTH_STD_MM,
                                     r_min=TissueProperties.RADIUS_BONE_DIAMETER_MEAN_MM / 2 - TissueProperties.RADIUS_BONE_DIAMETER_STD_MM / 2,
                                     r_max=TissueProperties.RADIUS_BONE_DIAMETER_MEAN_MM / 2 + TissueProperties.RADIUS_BONE_DIAMETER_STD_MM / 2)
    radius_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_bone_settings()
    radius_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BONE
    return radius_dict


def create_ulna_bone(relative_shift_mm=0.0):
    radius_dict = create_vessel_tube(x_min=relative_shift_mm - TissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM + TissueProperties.RADIUS_ULNA_BONE_SEPARATION_MEAN_MM,
                                     x_max=relative_shift_mm + TissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM + TissueProperties.RADIUS_ULNA_BONE_SEPARATION_MEAN_MM,
                                     z_min=TissueProperties.ULNA_BONE_DEPTH_MEAN_MM - TissueProperties.ULNA_BONE_DEPTH_STD_MM,
                                     z_max=TissueProperties.ULNA_BONE_DEPTH_MEAN_MM + TissueProperties.ULNA_BONE_DEPTH_STD_MM,
                                     r_min=TissueProperties.ULNA_BONE_DIAMETER_MEAN_MM / 2 - TissueProperties.ULNA_BONE_DIAMETER_STD_MM / 2,
                                     r_max=TissueProperties.ULNA_BONE_DIAMETER_MEAN_MM / 2 + TissueProperties.ULNA_BONE_DIAMETER_STD_MM / 2)
    radius_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_bone_settings()
    radius_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BONE
    return radius_dict


def create_subcutaneous_vein(relative_shift_mm=0.0):
    interosseous_dict = create_vessel_tube(x_min=-5 + relative_shift_mm, x_max=5 + relative_shift_mm,
                                           z_min=TissueProperties.SUBCUTANEOUS_VEIN_DEPTH_MEAN_MM - TissueProperties.SUBCUTANEOUS_VEIN_DEPTH_STD_MM,
                                           z_max=TissueProperties.SUBCUTANEOUS_VEIN_DEPTH_MEAN_MM + TissueProperties.SUBCUTANEOUS_VEIN_DEPTH_STD_MM,
                                           r_min=TissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_MEAN_MM / 2 - TissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_STD_MM / 2,
                                           r_max=TissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_MEAN_MM / 2 + TissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_STD_MM / 2)
    interosseous_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_venous_blood_settings()
    interosseous_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
    return interosseous_dict
