# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils.tags import Tags

EPS = 1e-20
"""
Defines the smallest increment that should be considered by SIMPA.
"""


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
    SOFT_TISSUE = 12
    LYMPH_NODE = 13


wavelength_dependent_properties = [
    Tags.DATA_FIELD_ABSORPTION_PER_CM,
    Tags.DATA_FIELD_SCATTERING_PER_CM,
    Tags.DATA_FIELD_ANISOTROPY,
    Tags.DATA_FIELD_REFRACTIVE_INDEX
]

wavelength_independent_properties = [
    Tags.DATA_FIELD_GRUNEISEN_PARAMETER,
    Tags.DATA_FIELD_SEGMENTATION,
    Tags.DATA_FIELD_OXYGENATION,
    Tags.DATA_FIELD_DENSITY,
    Tags.DATA_FIELD_SPEED_OF_SOUND,
    Tags.DATA_FIELD_ALPHA_COEFF
]

property_tags = wavelength_dependent_properties + wavelength_independent_properties

toolkit_tags = [Tags.KWAVE_PROPERTY_SENSOR_MASK, Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE]

simulation_output = [Tags.DATA_FIELD_FLUENCE,
                     Tags.DATA_FIELD_INITIAL_PRESSURE,
                     Tags.OPTICAL_MODEL_UNITS,
                     Tags.DATA_FIELD_TIME_SERIES_DATA,
                     Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                     Tags.DATA_FIELD_DIFFUSE_REFLECTANCE,
                     Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS,
                     Tags.DATA_FIELD_PHOTON_EXIT_POS,
                     Tags.DATA_FIELD_PHOTON_EXIT_DIR]

simulation_output_fields = [Tags.OPTICAL_MODEL_OUTPUT_NAME,
                            Tags.SIMULATION_PROPERTIES]
