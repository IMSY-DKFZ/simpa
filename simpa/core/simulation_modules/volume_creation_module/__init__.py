# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from simpa.utils.settings import Settings
from simpa.utils import Tags
from simpa.utils.tissue_properties import TissueProperties
import numpy as np
from simpa.core import SimulationModule
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling import save_hdf5
from simpa.utils.quality_assurance.data_sanity_testing import assert_equal_shapes, assert_array_well_defined


class VolumeCreatorModuleBase(SimulationModule):
    """
    Use this class to define your own volume creation adapter.

    """

    def __init__(self, global_settings: Settings):
        super(VolumeCreatorModuleBase, self).__init__(global_settings=global_settings)
        self.component_settings = global_settings.get_volume_creation_settings()

    def create_empty_volumes(self):
        volumes = dict()
        voxel_spacing = self.global_settings[Tags.SPACING_MM]
        volume_x_dim = int(round(self.global_settings[Tags.DIM_VOLUME_X_MM] / voxel_spacing))
        volume_y_dim = int(round(self.global_settings[Tags.DIM_VOLUME_Y_MM] / voxel_spacing))
        volume_z_dim = int(round(self.global_settings[Tags.DIM_VOLUME_Z_MM] / voxel_spacing))
        sizes = (volume_x_dim, volume_y_dim, volume_z_dim)

        for key in TissueProperties.property_tags:
            volumes[key] = np.zeros(sizes)

        return volumes, volume_x_dim, volume_y_dim, volume_z_dim

    @abstractmethod
    def create_simulation_volume(self) -> dict:
        """
        This method creates an in silico representation of a tissue as described in the settings file that is given.

        :return: A dictionary containing optical and acoustic properties as well as other characteristics of the
            simulated volume such as oxygenation, and a segmentation mask. All of these are given as 3d numpy arrays.
        :rtype: dict
        """
        pass

    def run(self, device):
        self.logger.info("VOLUME CREATION")

        volumes = self.create_simulation_volume()

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_equal_shapes(list(volumes.values()))
            for _volume_name in volumes.keys():
                if _volume_name == Tags.DATA_FIELD_OXYGENATION:
                    # oxygenation can have NaN by definition
                    continue
                assert_array_well_defined(volumes[_volume_name], array_name=_volume_name)

        save_volumes = dict()
        for key, value in volumes.items():
            if key in [Tags.DATA_FIELD_ABSORPTION_PER_CM, Tags.DATA_FIELD_SCATTERING_PER_CM,
                       Tags.DATA_FIELD_ANISOTROPY]:
                save_volumes[key] = {self.global_settings[Tags.WAVELENGTH]: value}
            else:
                save_volumes[key] = value

        volume_path = generate_dict_path(Tags.SIMULATION_PROPERTIES, self.global_settings[Tags.WAVELENGTH])
        save_hdf5(save_volumes, self.global_settings[Tags.SIMPA_OUTPUT_PATH], file_dictionary_path=volume_path)
