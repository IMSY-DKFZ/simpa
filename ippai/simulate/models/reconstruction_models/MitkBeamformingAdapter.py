from ippai.simulate import Tags
from ippai.simulate.models.reconstruction_models import ReconstructionAdapterBase
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

        beamforming_dict["MITK_beamforming"]["PA"]["Beamforming"]["@speedOfSound"] = settings[
            Tags.MEDIUM_SOUND_SPEED]
        beamforming_dict["MITK_beamforming"]["PA"]["Beamforming"]["@pitchInMeters"] = settings[
            Tags.SENSOR_ELEMENT_PITCH_CM]
        beamforming_dict["MITK_beamforming"]["PA"]["Beamforming"]["@reconstructionDepth"] = settings[
                                                                                                Tags.DIM_VOLUME_Z_MM] / 1000

        beamforming_dict["MITK_beamforming"]["PA"]["Resampling"]["@spacing"] = settings[Tags.SPACING_MM] / 2
        beamforming_dict["MITK_beamforming"]["PA"]["Resampling"]["@dimX"] = settings[Tags.SENSOR_NUM_ELEMENTS] * 2

        with open(save_path, "w") as xml_write_file:
            xmltodict.unparse(beamforming_dict, xml_write_file)

    def reconstruction_algorithm(self, time_series_sensor_data, settings):
        print("Calling MITK now........")

        tmp_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]
        tmp_input_path = tmp_path + "_input.nrrd"
        tmp_output_path = tmp_path + "_output.nrrd"
        tmp_settings_xml = tmp_path + "_settings.xml"

        with open(settings[Tags.RECONSTRUCTION_MITK_SETTINGS_XML], "r") as file:
            self.convert_settings_file(file, settings, tmp_settings_xml)

        time_series_sensor_data = np.atleast_3d(time_series_sensor_data)
        header = dict()
        header['space dimension'] = 3
        header['space directions'] = [[0.3, 0, 0], [0, 1/(settings[Tags.SENSOR_SAMPLING_RATE_MHZ]), 0], [0, 0, 1]]
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
        #os.remove(tmp_settings_xml)

        return reconstruction