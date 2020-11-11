# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
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

from simpa.core.device_digital_twins.pai_devices import PAIDeviceBase
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags, SegmentationClasses
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils.libraries.structure_library import HorizontalLayerStructure, Background
from simpa.utils.deformation_manager import get_functional_from_deformation_settings
import numpy as np


class MSOTPAIDevice(PAIDeviceBase):

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        pass

    def adjust_simulation_volume_and_settings(self, global_settings: Settings):

        sizes_mm = np.asarray([global_settings[Tags.DIM_VOLUME_X_MM],
                               global_settings[Tags.DIM_VOLUME_Y_MM],
                               global_settings[Tags.DIM_VOLUME_Z_MM]])

        probe_size_mm = 1 + 42.2
        mediprene_layer_height_mm = 1
        heavy_water_layer_height_mm = probe_size_mm - mediprene_layer_height_mm
        new_volume_height_mm = global_settings[Tags.DIM_VOLUME_Z_MM] + mediprene_layer_height_mm + \
                               heavy_water_layer_height_mm

        global_settings[Tags.DIM_VOLUME_Z_MM] = new_volume_height_mm

        for structure_key in global_settings[Tags.STRUCTURES]:
            structure_dict = global_settings[Tags.STRUCTURES][structure_key]
            if Tags.STRUCTURE_START_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_START_MM][2] = structure_dict[Tags.STRUCTURE_START_MM][2] + probe_size_mm
            if Tags.STRUCTURE_END_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_END_MM][2] = structure_dict[Tags.STRUCTURE_END_MM][2] + probe_size_mm

        mediprene_layer_settings = Settings({
            Tags.PRIORITY: 10,
            Tags.STRUCTURE_START_MM: [0, 0, heavy_water_layer_height_mm],
            Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm + mediprene_layer_height_mm],
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.mediprene()
        })

        mediprene_layer = HorizontalLayerStructure(global_settings, mediprene_layer_settings)
        global_settings[Tags.STRUCTURES]["mediprene"] = mediprene_layer.to_settings()

        background_settings = Settings({
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.heavy_water()
        })
        background = Background(global_settings, background_settings)
        global_settings[Tags.STRUCTURES][Tags.BACKGROUND] = background.to_settings()

        return global_settings

    def get_illuminator_definition(self):
        pass

    def get_detector_definition(self):
        pass