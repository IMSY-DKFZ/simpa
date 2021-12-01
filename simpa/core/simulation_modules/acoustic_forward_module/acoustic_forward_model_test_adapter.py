# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from simpa.utils import Tags
from simpa.core.simulation_modules.acoustic_forward_module import AcousticForwardModelBaseAdapter


class AcousticForwardModelTestAdapter(AcousticForwardModelBaseAdapter):

    def forward_model(self, device) -> np.ndarray:

        if Tags.ACOUSTIC_SIMULATION_3D in self.component_settings \
                and self.component_settings[Tags.ACOUSTIC_SIMULATION_3D]:
            return np.random.random((128, 128, 3000))
        else:
            return np.random.random((128, 3000))
