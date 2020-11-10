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
from simpa.utils.libraries.structure_library import HorizontalLayerStructure
from simpa.utils.deformation_manager import get_functional_from_deformation_settings
import numpy as np


class MSOTPAIDevice(PAIDeviceBase):

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        pass

    def adjust_simulation_volume_and_settings(self, simulation_volume_dict: dict, global_settings: Settings):
        wavelength = global_settings[Tags.WAVELENGTH]

        sizes_voxels = np.asarray(np.shape(simulation_volume_dict[Tags.PROPERTY_ABSORPTION_PER_CM]))
        sizes_mm = sizes_voxels * global_settings[Tags.SPACING_MM]

        deformation_adjustment_mm = 0
        if Tags.SIMULATE_DEFORMED_LAYERS in global_settings and global_settings[Tags.SIMULATE_DEFORMED_LAYERS]:
            x_positions_mm = np.linspace(0, sizes_mm[0], sizes_voxels[0])
            y_positions_mm = np.linspace(0, sizes_mm[1], sizes_voxels[1])
            functional = get_functional_from_deformation_settings(global_settings[Tags.DEFORMED_LAYERS_SETTINGS])
            deformation_adjustment_mm = np.min(functional(x_positions_mm, y_positions_mm))

        probe_size_mm = 1 + 42.2 #FIXME IS this even correct?
        probe_size_voxels = int(round(probe_size_mm / global_settings[Tags.SPACING_MM]))

        mediprene_layer_height_mm = 1
        mediprene_layer_height_voxels = int(round(mediprene_layer_height_mm / global_settings[Tags.SPACING_MM]))

        heavy_water_layer_height_mm = probe_size_mm - mediprene_layer_height_mm
        heavy_water_layer_height_voxels = probe_size_voxels - mediprene_layer_height_voxels # Deuterium is the rest

        new_volume_height_mm = global_settings[Tags.DIM_VOLUME_Z_MM] + mediprene_layer_height_mm + \
                               heavy_water_layer_height_mm
        new_volume_height_voxels = sizes_voxels[2] + mediprene_layer_height_voxels + heavy_water_layer_height_voxels
        # Fill all volumes to the desired height with None values.
        for key in simulation_volume_dict:
            new_volume = np.empty((sizes_voxels[0], sizes_voxels[1], new_volume_height_voxels)) * 1e-10
            new_volume[:, :, heavy_water_layer_height_voxels + mediprene_layer_height_voxels:] = simulation_volume_dict[key]
            simulation_volume_dict[key] = new_volume

        global_settings[Tags.DIM_VOLUME_Z_MM] = new_volume_height_mm

        mediprene_layer_settings = Settings({
            Tags.STRUCTURE_START_MM: [0, 0, heavy_water_layer_height_mm],
            Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm + mediprene_layer_height_mm],
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.mediprene()
        })

        # Add the mediprene layer (THIS SHOULD BE SUPER EFFICIENT)
        mediprene_layer = HorizontalLayerStructure(global_settings, mediprene_layer_settings)
        mediprene_indexes, _ = mediprene_layer.get_enclosed_indices(mediprene_layer.get_params_from_settings(mediprene_layer_settings))
        mediprene_properties = mediprene_layer.molecule_composition.get_properties_for_wavelength(wavelength)
        for key in simulation_volume_dict:
            simulation_volume_dict[key][mediprene_indexes] = mediprene_properties[key]

        heavy_water_molecular_composition = TISSUE_LIBRARY.heavy_water()
        hw_properties = heavy_water_molecular_composition.get_properties_for_wavelength(wavelength)

        # Add the heavy water layer (THIS IS THE SLOW PART)
        for x_idx in range(sizes_voxels[0]):
            for y_idx in range(sizes_voxels[1]):
                for z_idx in range(new_volume_height_voxels):
                    if simulation_volume_dict[Tags.PROPERTY_SEGMENTATION][
                                              x_idx, y_idx, z_idx] == SegmentationClasses.MEDIPRENE:
                        break
                    else:
                        for key in simulation_volume_dict:
                            simulation_volume_dict[key][x_idx, y_idx, z_idx] = hw_properties[key]

        return simulation_volume_dict, global_settings

    def get_illuminator_definition(self):
        pass

    def get_detector_definition(self):
        pass