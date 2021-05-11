"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import nrrd
import numpy as np
import subprocess
from simpa.utils import Tags
import os
from simpa.core.optical_simulation_module import OpticalForwardModuleBase


class OpticalForwardModelMcxyzAdapter(OpticalForwardModuleBase):

    def forward_model(self, absorption_cm, scattering_cm, anisotropy):

        shape = np.shape(absorption_cm)
        mcxyz_input = np.zeros((shape[0], shape[1], shape[2], 3))
        mcxyz_input[:, :, :, 0] = absorption_cm
        mcxyz_input[:, :, :, 1] = scattering_cm
        mcxyz_input[:, :, :, 2] = anisotropy

        tmp_path = self.global_settings[Tags.SIMULATION_PATH] + "/" + self.global_settings[Tags.VOLUME_NAME]
        tmp_input_path = tmp_path + "_input.nrrd"
        tmp_output_path = tmp_path + "_output.nrrd"
        nrrd.write(tmp_input_path, mcxyz_input, {'space dimension': 4,
                                                 'dimension': 4,
                                                 'space directions': [[self.global_settings[Tags.SPACING_MM]/10, 0,
                                                                       0, 0],
                                                             [0, self.global_settings[Tags.SPACING_MM]/10, 0, 0],
                                                             [0, 0, self.global_settings[Tags.SPACING_MM]/10, 0],
                                                             [0, 0, 0, 1]],
                                                 'mcflag': 4, 'launchflag': 0, 'boundaryflag': 1,
                                                 'launchPointX': 0, 'launchPointY': 0})

        cmd = list()
        cmd.append(self.component_settings[Tags.OPTICAL_MODEL_BINARY_PATH])
        cmd.append("-i")
        cmd.append(tmp_input_path)
        cmd.append("-o")
        cmd.append(tmp_output_path)
        if Tags.OPTICAL_MODEL_NUMBER_PHOTONS in self.component_settings:
            cmd.append("-n")
            cmd.append(str(self.component_settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS]))
        if Tags.OPTICAL_MODEL_ILLUMINATION_GEOMETRY_XML_FILE in self.component_settings:
            cmd.append("-p")
            cmd.append(str(self.component_settings[Tags.OPTICAL_MODEL_ILLUMINATION_GEOMETRY_XML_FILE]))

        subprocess.run(cmd)

        [fluence, _] = nrrd.read(tmp_output_path)

        os.remove(tmp_input_path)
        os.remove(tmp_output_path)

        return fluence
