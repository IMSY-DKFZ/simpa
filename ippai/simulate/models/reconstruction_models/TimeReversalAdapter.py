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

from ippai.utils import Tags
from ippai.simulate import SaveFilePaths
from ippai.simulate.models.reconstruction_models import ReconstructionAdapterBase
from ippai.io_handling.io_hdf5 import load_hdf5
import numpy as np
import scipy.io as sio
import subprocess
import os


class TimeReversalAdapter(ReconstructionAdapterBase):

    @staticmethod
    def get_acoustic_properties(settings, input_data):
        if Tags.PERFORM_UPSAMPLING in settings and settings[Tags.PERFORM_UPSAMPLING]:
            tmp_ac_properties = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], 
                                          SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.UPSAMPLED_DATA, 
                                                                                     settings[Tags.WAVELENGTH]))
        else:
            tmp_ac_properties = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH],
                                          SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.ORIGINAL_DATA,
                                                                                     settings[Tags.WAVELENGTH]))

        if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
            axes = (0, 1)
        else:
            axes = (0, 2)

        possible_acoustic_properties = [Tags.PROPERTY_SENSOR_MASK,
                                        Tags.PROPERTY_DIRECTIVITY_ANGLE,
                                        Tags.PROPERTY_SPEED_OF_SOUND,
                                        Tags.PROPERTY_DENSITY,
                                        Tags.PROPERTY_ALPHA_COEFF
                                        ]

        for acoustic_property in possible_acoustic_properties:
            if acoustic_property in tmp_ac_properties.keys():
                try:
                    input_data[acoustic_property] = np.rot90(tmp_ac_properties[acoustic_property], 3, axes=axes)
                except ValueError or KeyError:
                    print("{} not specified.".format(acoustic_property))

        return input_data

    def reconstruction_algorithm(self, time_series_sensor_data, settings):
        input_data = dict()
        input_data[Tags.TIME_SERIES_DATA] = time_series_sensor_data
        input_data = self.get_acoustic_properties(settings, input_data)
        acoustic_path = settings[Tags.IPPAI_OUTPUT_PATH] + ".mat"

        input_data["settings"] = settings
        sio.savemat(acoustic_path, input_data, long_field_names=True)

        cmd = list()
        cmd.append(settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
        cmd.append("-nodisplay")
        cmd.append("-nosplash")
        cmd.append("-r")
        cmd.append("addpath('" + settings[Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION] + "');" +
                   "time_reversal" + "('" + acoustic_path + "');exit;")
        cur_dir = os.getcwd()
        os.chdir(settings[Tags.SIMULATION_PATH])

        subprocess.run(cmd)

        reconstructed_data = sio.loadmat(acoustic_path + "tr.mat")["reconstructed_data"]

        return reconstructed_data
