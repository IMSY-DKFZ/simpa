"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from abc import abstractmethod
from simpa.utils.settings import Settings
from simpa.utils import Tags
from simpa.utils.tissue_properties import TissueProperties
import numpy as np
from simpa.core.simulation_components import SimulationModule
from simpa.core.device_digital_twins import DEVICE_MAP
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling import save_hdf5


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
        This method will be called to create a simulation volume.
        """
        pass

    def run(self):
        """
            This method is the main entry point of volume creation for the SIMPA framework.
            It uses the Tags.VOLUME_CREATOR tag to determine which of the volume creators
            should be used to create the simulation phantom.

            :param global_settings: the settings dictionary that contains the simulation instructions

            """
        self.logger.info("VOLUME CREATION")

        pa_device = None
        if Tags.DIGITAL_DEVICE in self.global_settings:
            try:
                pa_device = DEVICE_MAP[self.global_settings[Tags.DIGITAL_DEVICE]]
                pa_device.check_settings_prerequisites(self.global_settings)
            except KeyError as e:
                self.logger.warning(f"Intercepted exception during creation of PADevice instance: {str(e)}")
                pa_device = None

        if pa_device is not None:
            self.global_settings = pa_device.adjust_simulation_volume_and_settings(self.global_settings)

        volumes = self.create_simulation_volume()
        save_volumes = dict()
        for key, value in volumes.items():
            if key in [Tags.PROPERTY_ABSORPTION_PER_CM, Tags.PROPERTY_SCATTERING_PER_CM, Tags.PROPERTY_ANISOTROPY]:
                save_volumes[key] = {self.global_settings[Tags.WAVELENGTH]: value}
            else:
                save_volumes[key] = value

        volume_path = generate_dict_path(Tags.SIMULATION_PROPERTIES, self.global_settings[Tags.WAVELENGTH])
        save_hdf5(save_volumes, self.global_settings[Tags.SIMPA_OUTPUT_PATH], file_dictionary_path=volume_path)
