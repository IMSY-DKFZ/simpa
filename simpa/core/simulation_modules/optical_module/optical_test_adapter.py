# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.optical_module import OpticalAdapterBase
from simpa import Tags


class OpticalTestAdapter(OpticalAdapterBase):
    """
    This Adapter was created for testing purposes and only
    """

    def forward_model(self, absorption_cm, scattering_cm, anisotropy, illumination_geometry):
        results = {Tags.DATA_FIELD_FLUENCE: absorption_cm / ((1 - anisotropy) * scattering_cm)}
        return results
