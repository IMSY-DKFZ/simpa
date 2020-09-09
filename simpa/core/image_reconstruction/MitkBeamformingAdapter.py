# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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
import numpy as np
import subprocess
import xmltodict
import os
import nrrd


class MitkBeamformingAdapter(ReconstructionAdapterBase):

    def convert_settings_file(self, file, settings, save_path):
        settings_string = file.readlines()
        settings_string = "\n".join(settings_string)
        beamforming_dict = xmltodict.parse(settings_string)

        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@speedOfSoundMeterPerSecond"] = StandardProperties\
            .SPEED_OF_SOUND_GENERIC

        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@algorithm"] = settings[
            Tags.RECONSTRUCTION_ALGORITHM]

        beamforming_dict["ProcessingPipeline"]["PA"]["Beamforming"]["@pitchMilliMeter"] = settings[
            Tags.SENSOR_ELEMENT_PITCH_MM]

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
        beamforming_dict["ProcessingPipeline"]["PA"]["Resampling"]["@dimX"] = 70.856 / spacing

        with open(save_path, "w") as xml_write_file:
            xmltodict.unparse(beamforming_dict, xml_write_file, pretty=True, indent="\t")

    def reconstruction_algorithm(self, time_series_sensor_data, settings, distortion):
        print("Calling MITK now........")

        tmp_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]
        tmp_input_path = tmp_path + "_input.nrrd"
        tmp_output_path = tmp_path + "_output.nrrd"
        tmp_settings_xml = tmp_path + "_settings.xml"

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
