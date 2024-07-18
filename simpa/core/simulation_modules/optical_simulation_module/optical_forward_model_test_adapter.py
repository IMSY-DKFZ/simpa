# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.optical_simulation_module import OpticalForwardModuleBase
from simpa import Tags, Settings


class OpticalForwardModelTestAdapter(OpticalForwardModuleBase):
    """
    This Adapter was created for testing purposes and only
    """

    def get_default_component_settings(self) -> Settings:
        """
        :return: Loads default optical component settings 
        """

        default_settings = {}
        return Settings(default_settings)      

    def forward_model(self, absorption_cm, scattering_cm, anisotropy, illumination_geometry):
        results = {Tags.DATA_FIELD_FLUENCE: absorption_cm / ((1 - anisotropy) * scattering_cm)}
        return results
