# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
import struct
import subprocess
from simpa.utils import Tags
from simpa.core.simulation_modules.optical_simulation_module import OpticalForwardModuleBase
import json
import os
import gc


class MCXAdapter(OpticalForwardModuleBase):
    """
    This class implements a bridge to the mcx framework to integrate mcx into SIMPA.
    MCX is a GPU-enabled Monte-Carlo model simulation of photon transport in tissue::

        Fang, Qianqian, and David A. Boas. "Monte Carlo simulation of photon migration in 3D
        turbid media accelerated by graphics processing units."
        Optics express 17.22 (2009): 20178-20190.

    """

    def forward_model(self, absorption_cm, scattering_cm, anisotropy, illumination_geometry, probe_position_mm):
        if Tags.MCX_ASSUMED_ANISOTROPY in self.component_settings:
            _assumed_anisotropy = self.component_settings[Tags.MCX_ASSUMED_ANISOTROPY]
        else:
            _assumed_anisotropy = 0.9

        self.generate_mcx_input_file(absorption_cm=absorption_cm,
                                     scattering_cm=scattering_cm,
                                     anisotropy=_assumed_anisotropy,
                                     assumed_anisotropy=_assumed_anisotropy)

        mcx_volumetric_data_file = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                                   self.global_settings[Tags.VOLUME_NAME] + "_output"
        self.temporary_output.append(mcx_volumetric_data_file + '.mc2')
        self.mcx_volumetric_data_file = mcx_volumetric_data_file + '.mc2'

        # write settings to json
        # time = 1.16e-09
        # dt = 8e-12
        if Tags.TIME_STEP and Tags.TOTAL_TIME in self.component_settings:
            dt = self.component_settings[Tags.TIME_STEP]
            time = self.component_settings[Tags.TOTAL_TIME]
        else:
            time = 5e-09
            dt = 5e-09
        self.frames = int(time/dt)

        source = illumination_geometry.get_mcx_illuminator_definition(self.global_settings, probe_position_mm)

        settings_dict = {
            "Session": {
                "ID": mcx_volumetric_data_file,
                "DoAutoThread": 1,
                "Photons": self.component_settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS],
                "DoMismatch": 0
             },
            "Forward": {
                "T0": 0,
                "T1": time,
                "Dt": dt
            },
            "Optode": {
              "Source": source
            },
            "Domain": {
                "OriginType": 0,
                "LengthUnit": self.global_settings[Tags.SPACING_MM],
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
                        "g": _assumed_anisotropy,
                        "n": 1
                    }
                ],
                "MediaFormat": "muamus_float",
                "Dim": [self.nx, self.ny, self.nz],
                "VolumeFile": self.global_settings[Tags.SIMULATION_PATH] + "/" +
                              self.global_settings[Tags.VOLUME_NAME] + ".bin"
            }}

        if Tags.MCX_SEED not in self.component_settings:
            if Tags.RANDOM_SEED in self.global_settings:
                settings_dict["Session"]["RNGSeed"] = self.global_settings[Tags.RANDOM_SEED]
        else:
            settings_dict["Session"]["RNGSeed"] = self.component_settings[Tags.MCX_SEED]

        print(settings_dict)

        tmp_json_filename = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                            self.global_settings[Tags.VOLUME_NAME] + ".json"
        self.mcx_json_config_file = tmp_json_filename
        self.temporary_output.append(tmp_json_filename)
        with open(tmp_json_filename, "w") as json_file:
            json.dump(settings_dict, json_file, indent="\t")

        # run the simulation

        cmd = self.get_command()

        _ = subprocess.run(cmd)

        # Read output

        fluence = self.read_mcx_output()

        struct._clearcache()

        self.clean_mcx_output()

        return fluence

    def get_command(self):
        cmd = list()
        cmd.append(self.component_settings[Tags.OPTICAL_MODEL_BINARY_PATH])
        cmd.append("-f")
        cmd.append(self.mcx_json_config_file)
        cmd.append("-O")
        cmd.append("F")
        return cmd

    def generate_mcx_input_file(self, absorption_cm, scattering_cm, anisotropy, assumed_anisotropy):
        absorption_mm = absorption_cm / 10
        scattering_mm = scattering_cm / 10

        # FIXME Currently, mcx only accepts a single value for the anisotropy.
        #   In order to use the correct reduced scattering coefficient throughout the simulation,
        #   we adjust the scattering parameter to be more accurate in the diffuse regime.
        #   This will lead to errors, especially in the quasi-ballistic regime.

        given_reduced_scattering = (scattering_mm * (1 - anisotropy))
        scattering_mm = given_reduced_scattering / (1 - assumed_anisotropy)
        scattering_mm[scattering_mm < 1e-10] = 1e-10

        op_array = np.asarray([absorption_mm, scattering_mm])

        [_, self.nx, self.ny, self.nz] = np.shape(op_array)

        # create a binary of the volume

        optical_properties_list = list(np.reshape(op_array, op_array.size, "F"))
        del absorption_cm, absorption_mm, scattering_cm, scattering_mm, op_array
        gc.collect()
        mcx_input = struct.pack("f" * len(optical_properties_list), *optical_properties_list)
        del optical_properties_list
        gc.collect()
        tmp_input_path = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                         self.global_settings[Tags.VOLUME_NAME] + ".bin"
        self.temporary_output.append(tmp_input_path)
        with open(tmp_input_path, "wb") as input_file:
            input_file.write(mcx_input)

        del mcx_input, input_file
        struct._clearcache()
        gc.collect()

    def read_mcx_output(self):
        with open(self.mcx_volumetric_data_file, 'rb') as f:
            data = f.read()
        data = struct.unpack('%df' % (len(data) / 4), data)
        fluence = np.asarray(data).reshape([self.nx, self.ny, self.nz, self.frames], order='F')
        if np.shape(fluence)[3] == 1:
            fluence = np.squeeze(fluence, 3) * 100  # Convert from J/mm^2 to J/cm^2
        return fluence

    def clean_mcx_output(self):
        for f in self.temporary_output:
            if os.path.isfile(f):
                os.remove(f)
