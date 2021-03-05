# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
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

from abc import abstractmethod
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags
from simpa.utils.tissue_properties import TissueProperties
import numpy as np


class VolumeCreatorBase:
    """
    Use this class to define your own volume creation adapter.

    """

    def create_empty_volumes(self, global_settings):
        volumes = dict()
        voxel_spacing = global_settings[Tags.SPACING_MM]
        volume_x_dim = int(round(global_settings[Tags.DIM_VOLUME_X_MM] / voxel_spacing))
        volume_y_dim = int(round(global_settings[Tags.DIM_VOLUME_Y_MM] / voxel_spacing))
        volume_z_dim = int(round(global_settings[Tags.DIM_VOLUME_Z_MM] / voxel_spacing))
        sizes = (volume_x_dim, volume_y_dim, volume_z_dim)

        for key in TissueProperties.property_tags:
            volumes[key] = np.zeros(sizes)

        return volumes, volume_x_dim, volume_y_dim, volume_z_dim

    @abstractmethod
    def create_simulation_volume(self, settings: Settings) -> dict:
        """
        This method will be called to create a simulation volume.

        :param settings: the settings dictionary containing the simulation instructions.
        """
        pass
