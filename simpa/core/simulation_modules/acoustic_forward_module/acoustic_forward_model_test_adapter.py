# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from simpa.utils import Tags, Settings
from simpa.core.simulation_modules.acoustic_forward_module import AcousticForwardModelBaseAdapter


class AcousticForwardModelTestAdapter(AcousticForwardModelBaseAdapter):

    def get_default_component_settings(self) -> Settings:
        """
        :return: Loads default acoustic component settings 
        """

        default_settings = {}
        return Settings(default_settings)      

    def forward_model(self, device) -> np.ndarray:

        if Tags.ACOUSTIC_SIMULATION_3D in self.component_settings \
                and self.component_settings[Tags.ACOUSTIC_SIMULATION_3D]:
            return np.random.random((128, 128, 3000))
        else:
            return np.random.random((128, 3000))
