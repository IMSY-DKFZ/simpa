# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags, Settings
import numpy as np


class TissueProperties(dict):

    property_tags = [Tags.DATA_FIELD_ABSORPTION_PER_CM,
                     Tags.DATA_FIELD_SCATTERING_PER_CM,
                     Tags.DATA_FIELD_ANISOTROPY,
                     Tags.DATA_FIELD_GRUNEISEN_PARAMETER,
                     Tags.DATA_FIELD_SEGMENTATION,
                     Tags.DATA_FIELD_OXYGENATION,
                     Tags.DATA_FIELD_DENSITY,
                     Tags.DATA_FIELD_SPEED_OF_SOUND,
                     Tags.DATA_FIELD_ALPHA_COEFF]

    def __init__(self, settings: Settings):
        super().__init__()
        volume_x_dim, volume_y_dim, volume_z_dim = settings.get_volume_dimensions_voxels()
        self.volume_fraction = np.zeros((volume_x_dim, volume_y_dim, volume_z_dim))
        for key in TissueProperties.property_tags:
            self[key] = 0
