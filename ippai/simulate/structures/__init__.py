from ippai.simulate import Tags
from ippai.simulate.tissue_properties import get_epidermis_settings, get_dermis_settings, get_subcutaneous_fat_settings


def create_forearm_structures():
    structures_dict = dict()
    structures_dict["subcutaneous_fat"] = create_subcutaneous_fat_layer()
    structures_dict["dermis"] = create_dermis_layer()
    structures_dict["epidermis"] = create_epidermis_layer()
    return structures_dict


def create_epidermis_layer():
    epidermis_dict = dict()
    epidermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    epidermis_dict[Tags.STRUCTURE_LAYER_DEPTH_MIN] = 0
    epidermis_dict[Tags.STRUCTURE_LAYER_DEPTH_MAX] = 0
    epidermis_dict[Tags.STRUCTURE_LAYER_THICKNESS_MIN] = 0.06
    epidermis_dict[Tags.STRUCTURE_LAYER_THICKNESS_MAX] = 0.06
    epidermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_epidermis_settings()
    return epidermis_dict


def create_dermis_layer():
    dermis_dict = dict()
    dermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    dermis_dict[Tags.STRUCTURE_LAYER_DEPTH_MIN] = 0
    dermis_dict[Tags.STRUCTURE_LAYER_DEPTH_MAX] = 0
    dermis_dict[Tags.STRUCTURE_LAYER_THICKNESS_MIN] = 1.8
    dermis_dict[Tags.STRUCTURE_LAYER_THICKNESS_MAX] = 2.2
    dermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_dermis_settings()
    return dermis_dict


def create_subcutaneous_fat_layer():
    fat_dict = dict()
    fat_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    fat_dict[Tags.STRUCTURE_LAYER_DEPTH_MIN] = 1.5
    fat_dict[Tags.STRUCTURE_LAYER_DEPTH_MAX] = 1.5
    fat_dict[Tags.STRUCTURE_LAYER_THICKNESS_MIN] = 1.5
    fat_dict[Tags.STRUCTURE_LAYER_THICKNESS_MAX] = 1.9
    fat_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_subcutaneous_fat_settings()
    return fat_dict
