# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.reconstruction_module import ReconstructionAdapterBase
from simpa.utils import Settings


class ReconstructionModuleTestAdapter(ReconstructionAdapterBase):

    def get_default_component_settings(self) -> Settings:
        """
        :return: Loads default reconstruction component settings 
        """

        default_settings = {}
        return Settings(default_settings)      

    def reconstruction_algorithm(self, time_series_sensor_data, detection_geometry):
        return time_series_sensor_data / 10 + 5
