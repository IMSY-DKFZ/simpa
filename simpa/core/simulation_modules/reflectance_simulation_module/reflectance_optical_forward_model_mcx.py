"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
import logging
import numpy as np
import struct
import subprocess
import json
import jdata
import os
import gc
from typing import List, Dict

from simpa.utils import Tags
from simpa.core.simulation_modules.optical_simulation_module.optical_forward_model_mcx_adapter import MCXAdapter
from simpa.core.device_digital_twins.illumination_geometries.illumination_geometry_base import IlluminationGeometryBase


class ReflectanceMcxAdapter(MCXAdapter):
    """
    This class implements a bridge to the mcx framework to integrate mcx into SIMPA. This class targets specifically
    diffuse reflectance simulations
    MCX is a GPU-enabled Monte-Carlo model simulation of photon transport in tissue:

        Fang, Qianqian, and David A. Boas. "Monte Carlo simulation of photon migration in 3D
        turbid media accelerated by graphics processing units."
        Optics express 17.22 (2009): 20178-20190.

    """

    def forward_model(self,
                      absorption_cm: np.ndarray,
                      scattering_cm: np.ndarray,
                      anisotropy: np.ndarray,
                      illumination_geometry: IlluminationGeometryBase,
                      probe_position_mm: np.ndarray) -> np.ndarray:
        if Tags.MCX_ASSUMED_ANISOTROPY in self.component_settings:
            _assumed_anisotropy = self.component_settings[Tags.MCX_ASSUMED_ANISOTROPY]
        else:
            _assumed_anisotropy = 0.9

        res = self.generate_mcx_input_file(absorption_cm=absorption_cm,
                                           scattering_cm=scattering_cm,
                                           anisotropy=_assumed_anisotropy,
                                           assumed_anisotropy=_assumed_anisotropy)

        settings_dict = self.get_mcx_settings(illumination_geometry=illumination_geometry,
                                              probe_position_mm=probe_position_mm,
                                              assumed_anisotropy=_assumed_anisotropy,
                                              **res)

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

        self.remove_mcx_output()
        return fluence

    def get_mcx_settings(self,
                         illumination_geometry: IlluminationGeometryBase,
                         probe_position_mm: np.ndarray,
                         assumed_anisotropy: np.ndarray,
                         **kwargs) -> Dict:
        mcx_volumetric_data_file = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                                   self.global_settings[Tags.VOLUME_NAME] + "_output"
        self.mcx_volumetric_data_file = mcx_volumetric_data_file + '.jnii'
        self.mcx_photon_data_file = mcx_volumetric_data_file + '_detp.jdat'
        self.temporary_output.append(self.mcx_volumetric_data_file)
        self.temporary_output.append(self.mcx_photon_data_file)
        if Tags.TIME_STEP and Tags.TOTAL_TIME in self.component_settings:
            dt = self.component_settings[Tags.TIME_STEP]
            time = self.component_settings[Tags.TOTAL_TIME]
        else:
            time = 5e-09
            dt = 5e-09
        self.frames = int(time / dt)

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
                "Source": source,
                "Detector": self.get_detectors(kwargs.get(Tags.DATA_FIELD_VOLUME_SURFACE_ALONG_Z[0])),
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
                        "g": assumed_anisotropy,
                        "n": 1
                    }
                ],
                "MediaFormat": "muamus_float",
                "Dim": [self.nx, self.ny, self.nz],
                "VolumeFile": self.global_settings[Tags.SIMULATION_PATH] + "/" +
                              self.global_settings[Tags.VOLUME_NAME] + ".bin"
            }}
        return settings_dict

    def get_detectors(self, surface: np.ndarray) -> List:
        detectors = []
        if Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and \
                self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT]:
            for i in range(self.nx):
                for j in range(self.ny):
                    detector = {"Pos": [i, j, int(surface[i, j])],
                                "R": 1}
                    detectors.append(detector)
            detectors = []
        return detectors

    def get_command(self) -> List:
        cmd = list()
        cmd.append(self.component_settings[Tags.OPTICAL_MODEL_BINARY_PATH])
        cmd.append("-f")
        cmd.append(self.mcx_json_config_file)
        cmd.append("-O")
        cmd.append("F")
        cmd.append("-F")
        cmd.append("jnii")
        if Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and \
                self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT]:
            cmd.append("--bc")  # save photon exit position and direction
            cmd.append("______000010")
            cmd.append("--savedetflag")
            cmd.append("XV")
        if Tags.COMPUTE_DIFFUSE_REFLECTANCE in self.component_settings and \
                self.component_settings[Tags.COMPUTE_DIFFUSE_REFLECTANCE]:
            cmd.append("--saveref")  # save diffuse reflectance at 0 filled voxels outside of domain
        return cmd

    def generate_mcx_input_file(self,
                                absorption_cm: np.ndarray,
                                scattering_cm: np.ndarray,
                                anisotropy: np.ndarray,
                                assumed_anisotropy: np.ndarray) -> Dict:
        absorption_mm = absorption_cm / 10
        scattering_mm = scattering_cm / 10

        # FIXME Currently, mcx only accepts a single value for the anisotropy.
        #   In order to use the correct reduced scattering coefficient throughout the simulation,
        #   we adjust the scattering parameter to be more accurate in the diffuse regime.
        #   This will lead to errors, especially in the quasi-ballistic regime.

        given_reduced_scattering = (scattering_mm * (1 - anisotropy))
        scattering_mm = given_reduced_scattering / (1 - assumed_anisotropy)
        absorption_mm = self.pre_process_volumes(absorption_mm)
        scattering_mm = self.pre_process_volumes(scattering_mm)
        surface = self.get_volume_surface(absorption_mm)
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
        return {Tags.DATA_FIELD_VOLUME_SURFACE_ALONG_Z[0]: surface}

    def read_mcx_output(self):
        if os.path.isfile(self.mcx_volumetric_data_file) and self.mcx_volumetric_data_file.endswith('.jnii'):
            content = jdata.load(self.mcx_volumetric_data_file)
            fluence = content['NIFTIData']
        else:
            raise FileNotFoundError(f"Could not find .jnii file for {self.mcx_volumetric_data_file}")
        if Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and \
                self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT] and \
                os.path.isfile(self.mcx_photon_data_file):
            content = jdata.load(self.mcx_photon_data_file)
            photon_pos = content['MCXData']['PhotonData']['p']
            photon_dir = content['MCXData']['PhotonData']['v']
        return fluence

    @staticmethod
    def pre_process_volumes(array: np.ndarray) -> np.ndarray:
        if len(array.shape) == 3 and not np.all(array[:, :, 0] == 0):
            array = np.pad(array, ((0, 0), (0, 0), (1, 0)), "constant", constant_values=0)
        return array

    @staticmethod
    def get_volume_surface(array: np.ndarray) -> np.ndarray:
        return np.argmax(array, axis=-1) - 1