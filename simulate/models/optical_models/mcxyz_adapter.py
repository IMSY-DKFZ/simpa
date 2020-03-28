import nrrd
import numpy as np
import subprocess
from simulate import Tags
import os
from simulate.models.optical_models import OpticalForwardAdapterBase


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
        if Tags.OPTICAL_MODEL_PROBE_XML_FILE in settings:
            cmd.append("-p")
            cmd.append(str(settings[Tags.OPTICAL_MODEL_PROBE_XML_FILE]))

        subprocess.run(cmd)

        [fluence, _] = nrrd.read(tmp_output_path)

        os.remove(tmp_input_path)
        os.remove(tmp_output_path)

        return fluence




