"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.core.optical_simulation_module import OpticalForwardModuleBase


class OpticalForwardModelTestAdapter(OpticalForwardModuleBase):
    """
    This Adapter was created for tesing purposes and only
    """

    def forward_model(self, absorption_cm, scattering_cm, anisotropy):
        return absorption_cm / ((1 - anisotropy) * scattering_cm)
