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

import numpy as np
import struct
import subprocess
from simpa.utils import Tags
from simpa.core.optical_simulation import OpticalForwardAdapterBase
import json
import os
from simpa.core.optical_simulation.illumination_definition import define_illumination


class McxAdapter(OpticalForwardAdapterBase):
    """
    This class implements a bridge to the mcx framework to integrate mcx into SIMPA.
    MCX is a GPU-enabled Monte-Carlo model simulation of photon transport in tissue::

        Fang, Qianqian, and David A. Boas. "Monte Carlo simulation of photon migration in 3D
        turbid media accelerated by graphics processing units."
        Optics express 17.22 (2009): 20178-20190.

    """

    def forward_model(self, absorption_cm, scattering_cm, anisotropy, settings):

        absorption_mm = absorption_cm / 10
        scattering_mm = scattering_cm / 10
        op_array = np.asarray([absorption_mm, scattering_mm])

        [_, nx, ny, nz] = np.shape(op_array)

        # create a binary of the volume

        optical_properties_list = list(np.reshape(op_array, op_array.size, "F"))
        mcx_input = struct.pack("f" * len(optical_properties_list), *optical_properties_list)
        tmp_input_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+".bin"
        with open(tmp_input_path, "wb") as input_file:
            input_file.write(mcx_input)
        tmp_output_file = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+"_output"

        # write settings to json
        # time = 1.16e-09
        # dt = 8e-12
        if Tags.TIME_STEP and Tags.TOTAL_TIME in settings:
            dt = settings[Tags.TIME_STEP]
            time = settings[Tags.TOTAL_TIME]
        else:
            time = 5e-09
            dt = 5e-09
        frames = int(time/dt)

        if Tags.ILLUMINATION_TYPE in settings:
            source = define_illumination(settings, nx, ny, nz)
        else:
            source = {
                  "Pos": [
                      int(nx / 2) + 0.5, int(ny / 2) + 0.5, 1
                  ],
                  "Dir": [
                      0,
                      0.342027,
                      0.93969
                  ],
                  "Type": "pasetup",
                  "Param1": [
                      24.5 / settings[Tags.SPACING_MM],
                      0,
                      0,
                      22.8 / settings[Tags.SPACING_MM]
                  ],
                  "Param2": [
                      0,
                      0,
                      0,
                      0
                  ]
              }

        settings_dict = {
            "Session": {
                "ID": tmp_output_file,
                "DoAutoThread": 1,
                "Photons": settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS],
                "DoMismatch": 0
             },
            "Forward": {
                "T0": 0,
                "T1": time,
                "Dt": dt
            },
            # "Optode": {
            # 	"Source": {
            # 		"Pos": [int(nx/2)+0.5,int(ny/2)+0.5,1],
            # 		"Dir": [0,0,1]
            # 	}
            # },
            "Optode": {
              "Source": source
            },
            "Domain": {
                "OriginType": 0,
                "LengthUnit": settings[Tags.SPACING_MM],
                "Media": [
                    {
                        "mua": 0,
                        "mus": 0,
                        "g": 1,
                        "n": 1
                    },
                    {
                        "mua": 1,
                        "mus": 1,
                        "g": 0.9,
                        "n": 1
                    }
                ],
                "MediaFormat": "muamus_float",
                "Dim": [nx, ny, nz],
                "VolumeFile": settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+".bin"
            }}

        if Tags.MCX_SEED in settings:
            settings_dict["RNGSeed"] = settings[Tags.MCX_SEED]

        tmp_json_filename = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+".json"
        with open(tmp_json_filename, "w") as json_file:
            json.dump(settings_dict, json_file, indent="\t")

        # run the simulation

        cmd = list()
        cmd.append(settings[Tags.OPTICAL_MODEL_BINARY_PATH])
        cmd.append("-f")
        cmd.append(tmp_json_filename)
        cmd.append("-O")
        cmd.append("F")

        res = subprocess.run(cmd)

        print("TEST: ", res)

        # Read output

        with open(tmp_output_file+".mc2", 'rb') as f:
            data = f.read()
        data = struct.unpack('%df' % (len(data) / 4), data)
        fluence = np.asarray(data).reshape([nx, ny, nz, frames], order='F')
        if np.shape(fluence)[3] == 1:
            fluence = np.squeeze(fluence, 3) * 100  # Convert from J/mm^2 to J/cm^2

        os.remove(tmp_input_path)
        os.remove(tmp_output_file+".mc2")
        os.remove(tmp_json_filename)

        return fluence
