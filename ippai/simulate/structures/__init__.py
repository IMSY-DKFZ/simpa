from ippai.simulate import Tags
from ippai.simulate.tissue_properties import get_epidermis_settings, get_dermis_settings, \
    get_subcutaneous_fat_settings, get_blood_settings


def create_forearm_structures():
    structures_dict = dict()
    structures_dict["subcutaneous_fat"] = create_subcutaneous_fat_layer()
    structures_dict["dermis"] = create_dermis_layer()
    structures_dict["artery_1"] = create_vessel_tube()
    structures_dict["artery_2"] = create_vessel_tube()
    structures_dict["artery_3"] = create_vessel_tube()
    structures_dict["artery_4"] = create_vessel_tube()
    structures_dict["artery_5"] = create_vessel_tube()
    structures_dict["epidermis"] = create_epidermis_layer()
    return structures_dict


def create_epidermis_layer():
    epidermis_dict = dict()
    epidermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    epidermis_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 0
    epidermis_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 0
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = 0.06
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = 0.06
    epidermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_epidermis_settings()
    return epidermis_dict


def create_dermis_layer():
    dermis_dict = dict()
    dermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    dermis_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 0
    dermis_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 0
    dermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = 1.8
    dermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = 2.2
    dermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_dermis_settings()
    return dermis_dict


def create_subcutaneous_fat_layer():
    fat_dict = dict()
    fat_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    fat_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 1.5
    fat_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 1.5
    fat_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = 1.5
    fat_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = 1.9
    fat_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_subcutaneous_fat_settings()
    return fat_dict


def create_vessel_tube():
    vessel_dict = dict()
    vessel_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    vessel_dict[Tags.STRUCTURE_DEPTH_MIN_MM] = 1
    vessel_dict[Tags.STRUCTURE_DEPTH_MAX_MM] = 20
    vessel_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = 0.5
    vessel_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = 3
    vessel_dict[Tags.STRUCTURE_TUBE_START_X_MIN_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_START_X_MAX_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_START_Y_MIN_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_START_Y_MAX_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_START_Z_MIN_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_START_Z_MAX_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_END_X_MIN_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_END_X_MAX_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_END_Y_MIN_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_END_Y_MAX_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_END_Z_MIN_MM] = -1
    vessel_dict[Tags.STRUCTURE_TUBE_END_Z_MAX_MM] = -1
    vessel_dict[Tags.STRUCTURE_FORCE_ORTHAGONAL_TO_PLANE] = True
    vessel_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_blood_settings()
    return vessel_dict
