# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags, Settings
import numpy as np


class TissueProperties(dict):
    """
    The tissue properties contain a volumetric representation of each tissue parameter currently
    modelled in the SIMPA framework.

    It is a dictionary that is populated with each of the parameters.
    The values of the parameters can be either numbers or numpy arrays.
    It also contains a volume fraction field.
    """

    wavelength_dependent_properties = [
        Tags.DATA_FIELD_ABSORPTION_PER_CM,
        Tags.DATA_FIELD_SCATTERING_PER_CM,
        Tags.DATA_FIELD_ANISOTROPY
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

    def __init__(self, settings: Settings):
        super().__init__()
        volume_x_dim, volume_y_dim, volume_z_dim = settings.get_volume_dimensions_voxels()
        self.volume_fraction = np.zeros((volume_x_dim, volume_y_dim, volume_z_dim))
        for key in TissueProperties.property_tags:
            self[key] = 0
