# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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

import nrrd
import numpy as np
import subprocess
from simpa.utils import Tags
import os
from simpa.core.optical_simulation import OpticalForwardAdapterBase


class McxyzAdapter(OpticalForwardAdapterBase):

    def forward_model(self, absorption_cm, scattering_cm, anisotropy, settings):

        shape = np.shape(absorption_cm)
        mcxyz_input = np.zeros((shape[0], shape[1], shape[2], 3))
        mcxyz_input[:, :, :, 0] = absorption_cm
        mcxyz_input[:, :, :, 1] = scattering_cm
        mcxyz_input[:, :, :, 2] = anisotropy

        tmp_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]
        tmp_input_path = tmp_path + "_input.nrrd"
        tmp_output_path = tmp_path + "_output.nrrd"
        nrrd.write(tmp_input_path, mcxyz_input, {'space dimension': 4,
                                                 'dimension': 4,
                                                 'space directions': [[settings[Tags.SPACING_MM]/10, 0, 0, 0],
                                                             [0, settings[Tags.SPACING_MM]/10, 0, 0],
                                                             [0, 0, settings[Tags.SPACING_MM]/10, 0],
                                                             [0, 0, 0, 1]],
                                                 'mcflag': 4, 'launchflag': 0, 'boundaryflag': 1,
                                                 'launchPointX': 0, 'launchPointY': 0})

        cmd = list()
        cmd.append(settings[Tags.OPTICAL_MODEL_BINARY_PATH])
        cmd.append("-i")
        cmd.append(tmp_input_path)
        cmd.append("-o")
        cmd.append(tmp_output_path)
        if Tags.OPTICAL_MODEL_NUMBER_PHOTONS in settings:
            cmd.append("-n")
            cmd.append(str(settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS]))
        if Tags.OPTICAL_MODEL_ILLUMINATION_GEOMETRY_XML_FILE in settings:
            cmd.append("-p")
            cmd.append(str(settings[Tags.OPTICAL_MODEL_ILLUMINATION_GEOMETRY_XML_FILE]))

        subprocess.run(cmd)

        [fluence, _] = nrrd.read(tmp_output_path)

        os.remove(tmp_input_path)
        os.remove(tmp_output_path)

        return fluence




