# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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

from ippai.simulate.constants import SegmentationClasses, GeometryClasses
from ippai.utils import Tags
from ippai.utils import MorphologicalTissueProperties
from ippai.utils import TISSUE_LIBRARY
from ippai.utils.calculate import randomize


import numpy as np

def create_random_ellipse(x_min_mm=2, x_max_mm=35, depth_min_mm=3, depth_max_mm=18,
                          r_min_mm=0.5, r_max_mm=5.0,
                          eccentricity_min=0.25, eccentricity_max=3.5):
    rnd_tube_dict = dict()
    rnd_tube_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_ELLIPSE
    rnd_tube_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = depth_min_mm
    rnd_tube_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = depth_max_mm
    rnd_tube_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = r_min_mm
    rnd_tube_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = r_max_mm
    rnd_tube_dict[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] = x_min_mm
    rnd_tube_dict[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] = x_max_mm
    rnd_tube_dict[Tags.STRUCTURE_MIN_ECCENTRICITY] = eccentricity_min
    rnd_tube_dict[Tags.STRUCTURE_MAX_ECCENTRICITY] = eccentricity_max
    rnd_tube_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.blood_generic()
    rnd_tube_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
    return rnd_tube_dict


def create_random_background():
    rnd_bg_dict = dict()
    rnd_bg_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    # rnd_bg_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_random_background_settings()
    # rnd_bg_dict[Tags.STRUCTURE_USE_DISTORTION] = False
    # rnd_bg_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_BLOOD, Tags.KEY_OXY, Tags.KEY_WATER]
    # rnd_bg_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = 2
    rnd_bg_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.muscle()
    rnd_bg_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    return rnd_bg_dict


# #############################################
# RANDOM STRUCTURES
# #############################################
def create_random_structures():
    structures_dict = dict()
    structures_dict['background'] = create_random_background()
    structures_dict["dermis"] = create_dermis_layer()
    structures_dict["epidermis"] = create_epidermis_layer()
    num_ellipses = np.random.randint(3, 15)
    for i in range(num_ellipses):
            structures_dict["ellipse_" + str(i + 1)] = create_random_ellipse()
    return structures_dict


def create_vessel_tube(x_min=None, x_max=None, z_min=None, z_max=None, r_min=0.5, r_max=3.0):
    vessel_dict = dict()
    vessel_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    vessel_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = z_min
    vessel_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = z_max
    vessel_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = r_min
    vessel_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = r_max
    vessel_dict[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] = x_min
    vessel_dict[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] = x_max
    vessel_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.blood_generic()
    vessel_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
    return vessel_dict


# #############################################
# FOREARM MODEL STRUCTURES
# #############################################
def create_forearm_structures(relative_shift_mm=0, background_oxy=0.0, subcutaneous_vessel_spawn_probability=0.5):

    structures_dict = dict()
    structures_dict["muscle"] = create_muscle_background(background_oxy=background_oxy)
    structures_dict["dermis"] = create_dermis_layer(background_oxy=background_oxy)
    structures_dict["epidermis"] = create_epidermis_layer(background_oxy=background_oxy)
    structures_dict["radial_artery"] = create_radial_artery(relative_shift_mm)
    structures_dict["ulnar_artery"] = create_ulnar_artery(relative_shift_mm)
    structures_dict["interosseous_artery"] = create_interosseous_artery(relative_shift_mm)
    # structures_dict["radius"] = create_radius_bone(relative_shift_mm)
    # structures_dict["ulna"] = create_ulna_bone(relative_shift_mm)

    # for i in range(14):
    # for i in range(0):
    #     position = relative_shift_mm - 17.5 + i * 5
    #     if np.random.random() < subcutaneous_vessel_spawn_probability:
    #         structures_dict["subcutaneous_vessel_"+str(i+1)] = create_subcutaneous_vein(position)
    return structures_dict


def create_unrealistic_forearm_structures(relative_shift_mm=0, background_oxy=0.0,
                                          subcutaneous_vessel_spawn_probability=0.5,
                                          vessel_spawn_probability=0.1,
                                          radius_factor=1.5):
    structures_dict = dict()
    structures_dict["muscle"] = create_muscle_background(background_oxy=background_oxy)
    structures_dict["dermis"] = create_dermis_layer(background_oxy=background_oxy)
    structures_dict["epidermis"] = create_epidermis_layer(background_oxy=background_oxy)
    structures_dict["radial_artery"] = create_radial_artery(relative_shift_mm, radius_factor=radius_factor)
    structures_dict["ulnar_artery"] = create_ulnar_artery(relative_shift_mm, radius_factor=radius_factor)
    structures_dict["interosseous_artery"] = create_interosseous_artery(relative_shift_mm, radius_factor=radius_factor)
    structures_dict["radius"] = create_radius_bone(relative_shift_mm)
    structures_dict["ulna"] = create_ulna_bone(relative_shift_mm)

    for i in range(14):
        position = relative_shift_mm - 17.5 + i * 5
        if np.random.random() < subcutaneous_vessel_spawn_probability:
            structures_dict["subcutaneous_vessel_" + str(i + 1)] = create_subcutaneous_vein(position,
                                                                                            radius_factor=radius_factor)
    for i in range(5):
        if np.random.random() < vessel_spawn_probability:
            structures_dict["random_vessel" + str(i + 1)] = create_random_ellipse(eccentricity_min=0,
                                                                                  eccentricity_max=0.5)
    return structures_dict


def create_muscle_background():
    muscle_dict = dict()
    muscle_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    muscle_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.muscle()
    muscle_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.MUSCLE
    return muscle_dict

def create_background(mua=[1e-5], mus=[100.], g=[1.0], distortion=0., distortion_frequency=0.):
    background_dict = dict() 
    mua = mua[0]
    mus = mus[0]
    g = g[0]
    background_dict[Tags.STRUCTURE_USE_DISTORTION] = distortion
    background_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = distortion_frequency
    background_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    background_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
    background_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    background_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.BACKGROUND
    background_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    return background_dict


def create_epidermis_layer():
    epidermis_dict = dict()
    epidermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 0
    epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 0
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = MorphologicalTissueProperties.EPIDERMIS_THICKNESS_MEAN_MM - MorphologicalTissueProperties.EPIDERMIS_THICKNESS_STD_MM
    epidermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = MorphologicalTissueProperties.EPIDERMIS_THICKNESS_MEAN_MM + MorphologicalTissueProperties.EPIDERMIS_THICKNESS_STD_MM
    epidermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.epidermis()
    epidermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.EPIDERMIS
    return epidermis_dict


def create_dermis_layer(background_oxy=0.0):
    dermis_dict = dict()
    dermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 0
    dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 0
    dermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = MorphologicalTissueProperties.DERMIS_THICKNESS_MEAN_MM - MorphologicalTissueProperties.DERMIS_THICKNESS_STD_MM
    dermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = MorphologicalTissueProperties.DERMIS_THICKNESS_MEAN_MM + MorphologicalTissueProperties.DERMIS_THICKNESS_STD_MM
    dermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.dermis(background_oxy=background_oxy)
    dermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.DERMIS
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
    fat_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 1.5
    fat_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 1.5
    fat_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = 1.5
    fat_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = 1.9
    fat_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.subcutaneous_fat(background_oxy=background_oxy)
    fat_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.FAT
    return fat_dict


def create_radial_artery(relative_shift_mm=0.0, radius_factor=1.0):
    radial_dict = create_vessel_tube(x_min=relative_shift_mm + MorphologicalTissueProperties.RADIAL_ARTERY_X_POSITION_MEAN_MM - MorphologicalTissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                     x_max=relative_shift_mm + MorphologicalTissueProperties.RADIAL_ARTERY_X_POSITION_MEAN_MM + MorphologicalTissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                     z_min=MorphologicalTissueProperties.RADIAL_ARTERY_DEPTH_MEAN_MM - MorphologicalTissueProperties.RADIAL_ARTERY_DEPTH_STD_MM,
                                     z_max=MorphologicalTissueProperties.RADIAL_ARTERY_DEPTH_MEAN_MM + MorphologicalTissueProperties.RADIAL_ARTERY_DEPTH_STD_MM,
                                     r_min=radius_factor * (MorphologicalTissueProperties.RADIAL_ARTERY_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.RADIAL_ARTERY_DIAMETER_STD_MM / 2),
                                     r_max=radius_factor * (MorphologicalTissueProperties.RADIAL_ARTERY_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.RADIAL_ARTERY_DIAMETER_STD_MM / 2))
    radial_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.blood_arterial()
    radial_dict[Tags.CHILD_STRUCTURES] = dict()
    radial_dict[Tags.CHILD_STRUCTURES]["left_radial_accompanying_vein"] = create_vessel_tube(x_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                             x_max=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                             z_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                             z_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                             r_min=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2),
                                                                                             r_max=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2))
    radial_dict[Tags.CHILD_STRUCTURES]["left_radial_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        TISSUE_LIBRARY.blood_venous()
    radial_dict[Tags.CHILD_STRUCTURES]["right_radial_accompanying_vein"] = create_vessel_tube(x_min=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                              x_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                              z_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                              z_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                              r_min=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2),
                                                                                              r_max=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2))
    radial_dict[Tags.CHILD_STRUCTURES]["right_radial_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        TISSUE_LIBRARY.blood_venous()
    return radial_dict


def create_ulnar_artery(relative_shift_mm=0.0, radius_factor=1.0):
    ulnar_dict = create_vessel_tube(x_min=relative_shift_mm + MorphologicalTissueProperties.ULNAR_ARTERY_X_POSITION_MEAN_MM - MorphologicalTissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    x_max=relative_shift_mm + MorphologicalTissueProperties.ULNAR_ARTERY_X_POSITION_MEAN_MM + MorphologicalTissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    z_min=MorphologicalTissueProperties.ULNAR_ARTERY_DEPTH_MEAN_MM - MorphologicalTissueProperties.ULNAR_ARTERY_DEPTH_STD_MM,
                                    z_max=MorphologicalTissueProperties.ULNAR_ARTERY_DEPTH_MEAN_MM + MorphologicalTissueProperties.ULNAR_ARTERY_DEPTH_STD_MM,
                                    r_min=radius_factor * (MorphologicalTissueProperties.ULNAR_ARTERY_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.ULNAR_ARTERY_DIAMETER_STD_MM / 2),
                                    r_max=radius_factor * (MorphologicalTissueProperties.ULNAR_ARTERY_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.ULNAR_ARTERY_DIAMETER_STD_MM / 2))
    ulnar_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.blood_arterial()
    ulnar_dict[Tags.CHILD_STRUCTURES] = dict()
    ulnar_dict[Tags.CHILD_STRUCTURES]["left_ulnar_accompanying_vein"] = create_vessel_tube(x_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                           x_max=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                           z_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                           z_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                           r_min=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2),
                                                                                           r_max=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2))
    ulnar_dict[Tags.CHILD_STRUCTURES]["left_ulnar_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        TISSUE_LIBRARY.blood_venous()
    ulnar_dict[Tags.CHILD_STRUCTURES]["right_ulnar_accompanying_vein"] = create_vessel_tube(x_min=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM - MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                            x_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_MEAN_MM + MorphologicalTissueProperties.ACCOMPANYING_VEIN_DISTANCE_STD_MM),
                                                                                            z_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                            z_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                            r_min=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2),
                                                                                            r_max=radius_factor * (MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.RADIAL_VEIN_DIAMETER_STD_MM / 2))
    ulnar_dict[Tags.CHILD_STRUCTURES]["right_ulnar_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        TISSUE_LIBRARY.blood_venous()
    return ulnar_dict


def create_interosseous_artery(relative_shift_mm=0.0, radius_factor=1.0):
    inter_dict = create_vessel_tube(x_min=relative_shift_mm + MorphologicalTissueProperties.MEDIAN_ARTERY_X_POSITION_MEAN_MM - MorphologicalTissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    x_max=relative_shift_mm + MorphologicalTissueProperties.MEDIAN_ARTERY_X_POSITION_MEAN_MM + MorphologicalTissueProperties.ARTERY_X_POSITION_UNCERTAINTY_MM,
                                    z_min=MorphologicalTissueProperties.MEDIAN_ARTERY_DEPTH_MEAN_MM - MorphologicalTissueProperties.MEDIAN_ARTERY_DEPTH_STD_MM,
                                    z_max=MorphologicalTissueProperties.MEDIAN_ARTERY_DEPTH_MEAN_MM + MorphologicalTissueProperties.MEDIAN_ARTERY_DEPTH_STD_MM,
                                    r_min=radius_factor * (MorphologicalTissueProperties.MEDIAN_ARTERY_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.MEDIAN_ARTERY_DIAMETER_STD_MM / 2),
                                    r_max=radius_factor * (MorphologicalTissueProperties.MEDIAN_ARTERY_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.MEDIAN_ARTERY_DIAMETER_STD_MM / 2))
    inter_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.blood_arterial()
    inter_dict[Tags.CHILD_STRUCTURES] = dict()
    inter_dict[Tags.CHILD_STRUCTURES]["left_inter_accompanying_vein"] = create_vessel_tube(x_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM - MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM),
                                                                                           x_max=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM + MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM),
                                                                                           z_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                           z_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                           r_min=radius_factor * (MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2),
                                                                                           r_max=radius_factor * (MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2))
    inter_dict[Tags.CHILD_STRUCTURES]["left_inter_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        TISSUE_LIBRARY.blood_venous()
    inter_dict[Tags.CHILD_STRUCTURES]["right_inter_accompanying_vein"] = create_vessel_tube(x_min=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM - MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM),
                                                                                            x_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM + MorphologicalTissueProperties.ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM),
                                                                                            z_min=radius_factor * (- MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                            z_max=radius_factor * (MorphologicalTissueProperties.ACCOMPANYING_VEIN_DEPTH_STD_MM / 2),
                                                                                            r_min=radius_factor * (MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2),
                                                                                            r_max=radius_factor * (MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2))
    inter_dict[Tags.CHILD_STRUCTURES]["right_inter_accompanying_vein"][Tags.STRUCTURE_TISSUE_PROPERTIES] = \
        TISSUE_LIBRARY.blood_venous()
    return inter_dict


def create_radius_bone(relative_shift_mm=0.0):
    radius_dict = create_vessel_tube(x_min=relative_shift_mm - MorphologicalTissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM,
                                     x_max=relative_shift_mm + MorphologicalTissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM,
                                     z_min=MorphologicalTissueProperties.RADIUS_BONE_DEPTH_MEAN_MM - MorphologicalTissueProperties.RADIUS_BONE_DEPTH_STD_MM,
                                     z_max=MorphologicalTissueProperties.RADIUS_BONE_DEPTH_MEAN_MM + MorphologicalTissueProperties.RADIUS_BONE_DEPTH_STD_MM,
                                     r_min=MorphologicalTissueProperties.RADIUS_BONE_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.RADIUS_BONE_DIAMETER_STD_MM / 2,
                                     r_max=MorphologicalTissueProperties.RADIUS_BONE_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.RADIUS_BONE_DIAMETER_STD_MM / 2)
    radius_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.bone()
    radius_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BONE
    return radius_dict


def create_ulna_bone(relative_shift_mm=0.0):
    radius_dict = create_vessel_tube(x_min=relative_shift_mm - MorphologicalTissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM + MorphologicalTissueProperties.RADIUS_ULNA_BONE_SEPARATION_MEAN_MM,
                                     x_max=relative_shift_mm + MorphologicalTissueProperties.RADIUS_ULNA_BONE_POSITION_STD_MM + MorphologicalTissueProperties.RADIUS_ULNA_BONE_SEPARATION_MEAN_MM,
                                     z_min=MorphologicalTissueProperties.ULNA_BONE_DEPTH_MEAN_MM - MorphologicalTissueProperties.ULNA_BONE_DEPTH_STD_MM,
                                     z_max=MorphologicalTissueProperties.ULNA_BONE_DEPTH_MEAN_MM + MorphologicalTissueProperties.ULNA_BONE_DEPTH_STD_MM,
                                     r_min=MorphologicalTissueProperties.ULNA_BONE_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.ULNA_BONE_DIAMETER_STD_MM / 2,
                                     r_max=MorphologicalTissueProperties.ULNA_BONE_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.ULNA_BONE_DIAMETER_STD_MM / 2)
    radius_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.bone()
    radius_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BONE
    return radius_dict


def create_subcutaneous_vein(relative_shift_mm=0.0, radius_factor=1.0):
    interosseous_dict = create_vessel_tube(x_min=-5 + relative_shift_mm, x_max=5 + relative_shift_mm,
                                           z_min=MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_MEAN_MM - MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_STD_MM,
                                           z_max=MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_MEAN_MM + MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_STD_MM,
                                           r_min=radius_factor * (MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_MEAN_MM / 2 - MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_STD_MM / 2),
                                           r_max=radius_factor * (MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_MEAN_MM / 2 + MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_STD_MM / 2))
    interosseous_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.blood_venous()
    interosseous_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
    return interosseous_dict



def create_tube(mua=[10.], mus=[150.], g=[1.], distortion=0., distortion_frequency=0.,            structure_center_depth_mm=10., structure_radius_mm=5., structure_center_x_mm=0., structure_center_z_mm=10.):
    print(mua)
    tube_dict = dict()
    mua = mua[0].round()
    mus = mus[0]
    g = g[0] 
    tube_dict[Tags.STRUCTURE_USE_DISTORTION] = distortion
    tube_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = distortion_frequency
    tube_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = structure_center_depth_mm
    tube_dict[Tags.STRUCTURE_RADIUS_MM] = structure_radius_mm
    tube_dict[Tags.STRUCTURE_CENTER_X_MM] = structure_center_x_mm
    tube_dict[Tags.STRUCTURE_CENTER_Z_MM] = structure_center_z_mm
    tube_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
    tube_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    tube_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    tube_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    tube_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.TUBE
    return tube_dict


def create_sphere(mua=[10.], mus=[150.], g=[1.], distortion=0., distortion_frequency=0.,            structure_center_depth_mm=10., structure_radius_mm=5., structure_center_x_mm=0., structure_center_y_mm=10.):
    sphere_dict = dict()
    mua = mua[0]
    mus = mus[0]
    g = g[0]
    sphere_dict[Tags.STRUCTURE_USE_DISTORTION] = distortion
    sphere_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = distortion_frequency
    sphere_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = structure_center_depth_mm
    sphere_dict[Tags.STRUCTURE_RADIUS_MM] = structure_radius_mm
    sphere_dict[Tags.STRUCTURE_CENTER_X_MM] = structure_center_x_mm
    sphere_dict[Tags.STRUCTURE_CENTER_Y_MM] = structure_center_y_mm
    sphere_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
    sphere_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    sphere_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_SPHERE
    sphere_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    sphere_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.SPHERE
    return sphere_dict


def create_layer(mua=[10.], mus=[150.], g=[1.0], distortion=0., distortion_frequency=0,                    structure_center_depth_mm=5., structure_thickness_mm=0.5):
    layer_dict = dict()
    mua=mua[0]
    mus=mus[0]
    g=g[0]
    layer_dict[Tags.STRUCTURE_USE_DISTORTION] = distortion
    layer_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = distortion_frequency
    layer_dict[Tags.STRUCTURE_THICKNESS_MM] = structure_thickness_mm
    layer_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = structure_center_depth_mm
    layer_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
    layer_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    layer_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
    layer_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.DERMIS
    layer_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.LAYER
    return layer_dict


def create_pyramid(mua=[10.], mus=[100.], g=[1.0], distortion=0., distortion_frequency=0., structure_center_depth_mm=5., structure_center_x_mm=0., structure_center_y_mm=0., strucutre_height_mm=5., structure_basis_extent=3., structure_pyramid_orientation=0.):
    pyramid_dict = dict()
    mua =  mua[0]
    mus = mus[0]
    g = g[0] 
    pyramid_dict[Tags.STRUCTURE_USE_DISTORTION] = distortion
    pyramid_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = distortion_frequency
    pyramid_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = structure_center_depth_mm
    pyramid_dict[Tags.STRUCTURE_CENTER_X_MM] = structure_center_x_mm
    pyramid_dict[Tags.STRUCTURE_CENTER_Y_MM] = structure_center_y_mm
    pyramid_dict[Tags.STRUCTURE_HEIGHT_MM] = strucutre_height_mm
    pyramid_dict[Tags.STRUCTURE_BASIS_EXTENT_MM] = structure_basis_extent
    pyramid_dict[Tags.STRUCTURE_PYRAMID_ORIENTATION] = structure_pyramid_orientation
    #pyramid_dict[Tags.STRUCTURE_PYRAMID_ORIENTATION_XY] = settings_dict["randomness"] * (settings_dict["pyramid_orientation_xy"][0] - settings_dict["pyramid_orientation_xy"][1]) + settings_dict["pyramid_orientation_xy"][0]
    #pyramid_dict[Tags.STRUCTURE_PYRAMID_ORIENTATION_XZ] = settings_dict["randomness"] * (settings_dict["pyramid_orientation_xz"][0] - settings_dict["pyramid_orientation_xz"][1]) + settings_dict["pyramid_orientation_xz"][0]
    #pyramid_dict[Tags.STRUCTURE_PYRAMID_ORIENTATION_YZ] = settings_dict["randomness"] * (settings_dict["pyramid_orientation_yz"][0] - settings_dict["pyramid_orientation_yz"][1]) + settings_dict["pyramid_orientation_yz"][0]
    pyramid_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
    pyramid_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    pyramid_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_PYRAMID
    pyramid_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    pyramid_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.PYRAMID
    return pyramid_dict

def create_cube(mua=[10.], mus=[100.], g=[1.0], distortion=0., distortion_frequency=0., structure_center_y_mm=0., structure_center_x_mm=0., structure_center_z_mm=5.,structure_length_x_mm=2., structure_length_y_mm=2., structure_length_z_mm=2.):
    cube_dict = dict()
    mua = mua[0]
    mus = mus[0]
    g = g[0]
    cube_dict[Tags.STRUCTURE_USE_DISTORTION] = distortion
    cube_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = distortion_frequency
    cube_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = structure_center_z_mm
    cube_dict[Tags.STRUCTURE_CENTER_X_MM] = structure_center_x_mm
    cube_dict[Tags.STRUCTURE_CENTER_Y_MM] = structure_center_y_mm
    cube_dict[Tags.STRUCTURE_LENGTH_X_MM] = structure_length_x_mm
    cube_dict[Tags.STRUCTURE_LENGTH_Y_MM] = structure_length_y_mm
    cube_dict[Tags.STRUCTURE_LENGTH_Z_MM] = structure_length_z_mm
    cube_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
    cube_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    cube_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_CUBE
    cube_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    cube_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.CUBE
    return cube_dict

def create_cubical_tube(mua=[10.], mus=[100.], g=[1.0], distortion=0., distortion_frequency=0., structure_center_z_mm=5., structure_radius_mm=0.5, structure_center_x_mm=0.):
    cubical_tube_dict = dict()
    mua = mua[0] 
    mus = mus[0]
    g = g[0]
    cubical_tube_dict[Tags.STRUCTURE_USE_DISTORTION] = distortion
    cubical_tube_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = distortion_frequency
    cubical_tube_dict[Tags.STRUCTURE_CENTER_Z_MM] = structure_center_z_mm
    cubical_tube_dict[Tags.STRUCTURE_RADIUS_MM] = structure_radius_mm
    cubical_tube_dict[Tags.STRUCTURE_CENTER_X_MM] = structure_center_x_mm
    cubical_tube_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
    cubical_tube_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    cubical_tube_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_CUBICAL_TUBE
    cubical_tube_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    cubical_tube_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.CUBICAL_TUBE
    return cubical_tube_dict