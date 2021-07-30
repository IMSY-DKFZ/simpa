"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags


EPS = 1e-20
"""
Defines the smallest increment that should be considered by SIMPA.
"""


class SaveFilePaths:
    """
    The save file paths specify the path of a specific data structure in the dictionary of the simpa output hdf5.
    All of these paths have to be used like:
    SaveFilePaths.PATH + "data_structure"
    """
    SIMULATION_PROPERTIES = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/"
    OPTICAL_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/"
    ACOUSTIC_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.TIME_SERIES_DATA + "/"
    NOISE_ACOUSTIC_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.TIME_SERIES_DATA_NOISE + "/"
    RECONSTRCTION_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.RECONSTRUCTED_DATA + "/"
    NOISE_RECONSTRCTION_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.RECONSTRUCTED_DATA_NOISE + "/"


class SegmentationClasses:
    """
    The segmentation classes define which "tissue types" are modelled in the simulation volumes.
    """
    GENERIC = -1
    AIR = 0
    MUSCLE = 1
    BONE = 2
    BLOOD = 3
    EPIDERMIS = 4
    DERMIS = 5
    FAT = 6
    ULTRASOUND_GEL = 7
    WATER = 8
    HEAVY_WATER = 9
    COUPLING_ARTIFACT = 10
    MEDIPRENE = 11
