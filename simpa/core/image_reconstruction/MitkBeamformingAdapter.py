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

from simpa.utils import Tags
from simpa.utils import StandardProperties
from simpa.core.image_reconstruction import ReconstructionAdapterBase
from simpa.core.device_digital_twins.msot_devices import MSOTAcuityEcho
import numpy as np
import subprocess
import xmltodict
import os
import nrrd


class MitkBeamformingAdapter(ReconstructionAdapterBase):
    """
    This adapter can be used to reconstruct an image with the MITK beamforming tool.
    However, there is only support for linear and curved transducers.
    """

    def convert_settings_file(self, file, settings, save_path):

        if Tags.DIGITAL_DEVICE in settings and settings[Tags.DIGITAL_DEVICE] == Tags.DIGITAL_DEVICE_MSOT:
            PA_device = MSOTAcuityEcho()
        else:
            # default settings for now
            PA_device = MSOTAcuityEcho()

        settings_string = file.readlines()
        settings_string = "\n".join(settings_string)
        beamforming_dict = xmltodict.parse(settings_string)

        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@speedOfSoundMeterPerSecond"] = StandardProperties\
            .SPEED_OF_SOUND_GENERIC

        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@algorithm"] = settings[
            Tags.RECONSTRUCTION_ALGORITHM]

        if settings[Tags.SENSOR_NUM_USED_ELEMENTS] < PA_device.number_detector_elements:
            pitch = PA_device.pitch_mm*PA_device.number_detector_elements/settings[Tags.SENSOR_NUM_USED_ELEMENTS]
        else:
            pitch = PA_device.pitch_mm
        del settings[Tags.SENSOR_NUM_USED_ELEMENTS]
        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@pitchMilliMeter"] = pitch

        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@reconstructionDepthMeter"] = 0.08

        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@reconstructedXDimension"] = 256

        beamforming_dict["ProcessingPipeline"]["PA"]["BMode"]["@method"] = settings[Tags.RECONSTRUCTION_BMODE_METHOD]

        beamforming_dict["ProcessingPipeline"]["PA"]["Cropping"]["@do"] = 0

        beamforming_dict["ProcessingPipeline"]["PA"]["Resampling"]["@do"] = 1
        spacing = settings[Tags.SPACING_MM]
        if Tags.PERFORM_UPSAMPLING in settings:
            if settings[Tags.PERFORM_UPSAMPLING]:
                spacing = spacing / settings[Tags.UPSCALE_FACTOR]

        beamforming_dict["ProcessingPipeline"]["PA"]["Resampling"]["@spacing"] = spacing
        beamforming_dict["ProcessingPipeline"]["PA"]["Resampling"]["@dimX"] = PA_device.probe_width_mm / spacing

        with open(save_path, "w") as xml_write_file:
            xmltodict.unparse(beamforming_dict, xml_write_file, pretty=True, indent="\t")

    def reconstruction_algorithm(self, time_series_sensor_data, settings):
        print("Calling MITK now........")

        tmp_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]
        tmp_input_path = tmp_path + "_input.nrrd"
        tmp_output_path = tmp_path + "_output.nrrd"
        tmp_settings_xml = tmp_path + "_settings.xml"

        settings[Tags.SENSOR_NUM_USED_ELEMENTS] = np.shape(time_series_sensor_data)[0]

        with open(settings[Tags.RECONSTRUCTION_MITK_SETTINGS_XML], "r") as file:
            self.convert_settings_file(file, settings, tmp_settings_xml)

        time_series_sensor_data = np.atleast_3d(time_series_sensor_data)

        upscale_factor = 1
        if Tags.PERFORM_UPSAMPLING in settings:
            if settings[Tags.PERFORM_UPSAMPLING]:
                upscale_factor = settings[Tags.UPSCALE_FACTOR]

        header = dict()
        header['space dimension'] = 3
        header['space directions'] = [[settings[Tags.SPACING_MM]/upscale_factor, 0, 0],
                                      [0, settings["dt_acoustic_sim"]*10**6, 0],
                                      [0, 0, 1]]
        nrrd.write(tmp_input_path, time_series_sensor_data, header)

        cmd = list()
        cmd.append(settings[Tags.RECONSTRUCTION_MITK_BINARY_PATH])
        cmd.append("-i")
        cmd.append(tmp_input_path)
        cmd.append("-o")
        cmd.append(tmp_output_path)
        cmd.append("-s")
        cmd.append(tmp_settings_xml)
        cmd.append("-t")
        cmd.append("PA")

        subprocess.run(cmd)

        reconstruction, _ = nrrd.read(tmp_output_path)

        os.remove(tmp_input_path)
        os.remove(tmp_output_path)
        os.remove(tmp_settings_xml)

        return reconstruction
