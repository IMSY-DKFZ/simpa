# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
import struct
import subprocess
from simpa.utils import Tags, Settings
from simpa.core.simulation_modules.optical_simulation_module import OpticalForwardModuleBase
from simpa.core.device_digital_twins.illumination_geometries.illumination_geometry_base import IlluminationGeometryBase
import json
import os
import gc
from typing import List, Dict, Tuple


class MCXAdapter(OpticalForwardModuleBase):
    """
    This class implements a bridge to the mcx framework to integrate mcx into SIMPA. This adapter only allows for
    computation of fluence, for computations of diffuse reflectance, take a look at `simpa.ReflectanceMcxAdapter`

    .. note::
        MCX is a GPU-enabled Monte-Carlo model simulation of photon transport in tissue:
        Fang, Qianqian, and David A. Boas. "Monte Carlo simulation of photon migration in 3D
        turbid media accelerated by graphics processing units."
        Optics express 17.22 (2009): 20178-20190.

    """

    def __init__(self, global_settings: Settings):
        """
        initializes MCX-specific configuration and clean-up instances

        :param global_settings: global settings used during simulations
        """
        super(MCXAdapter, self).__init__(global_settings=global_settings)
        self.mcx_json_config_file = None
        self.mcx_volumetric_data_file = None
        self.frames = None
        self.mcx_output_suffixes = {'mcx_volumetric_data_file': '.mc2'}

    def forward_model(self,
                      absorption_cm: np.ndarray,
                      scattering_cm: np.ndarray,
                      anisotropy: np.ndarray,
                      illumination_geometry: IlluminationGeometryBase) -> Dict:
        """
        runs the MCX simulations. Binary file containing scattering and absorption volumes is temporarily created as
        input for MCX. A JSON serializable file containing the configuration required by MCx is also generated.
        The set of flags parsed to MCX is built based on the Tags declared in `self.component_settings`, the results
        from MCX are used to populate an instance of Dict and returned.

        :param absorption_cm: array containing the absorption of the tissue in `cm` units
        :param scattering_cm: array containing the scattering of the tissue in `cm` units
        :param anisotropy: array containing the anisotropy of the volume defined by `absorption_cm` and `scattering_cm`
        :param illumination_geometry: and instance of `IlluminationGeometryBase` defining the illumination geometry
        :return: `Dict` containing the results of optical simulations, the keys in this dictionary-like object
            depend on the Tags defined in `self.component_settings`
        """
        if Tags.MCX_ASSUMED_ANISOTROPY in self.component_settings:
            _assumed_anisotropy = self.component_settings[Tags.MCX_ASSUMED_ANISOTROPY]
        else:
            _assumed_anisotropy = 0.9

        self.generate_mcx_bin_input(absorption_cm=absorption_cm,
                                    scattering_cm=scattering_cm,
                                    anisotropy=_assumed_anisotropy,
                                    assumed_anisotropy=_assumed_anisotropy)

        settings_dict = self.get_mcx_settings(illumination_geometry=illumination_geometry,
                                              assumed_anisotropy=_assumed_anisotropy)

        print(settings_dict)
        self.generate_mcx_json_input(settings_dict=settings_dict)
        # run the simulation
        cmd = self.get_command()
        self.run_mcx(cmd)

        # Read output
        results = self.read_mcx_output()

        struct._clearcache()

        # clean temporary files
        self.remove_mcx_output()
        return results

    def generate_mcx_json_input(self, settings_dict: Dict) -> None:
        """
        generates JSON serializable file with settings needed by MCX to run simulations.

        :param settings_dict: dictionary to be saved as .json
        :return: None
        """
        tmp_json_filename = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                            self.global_settings[Tags.VOLUME_NAME] + ".json"
        self.mcx_json_config_file = tmp_json_filename
        self.temporary_output_files.append(tmp_json_filename)
        with open(tmp_json_filename, "w") as json_file:
            json.dump(settings_dict, json_file, indent="\t")

    def get_mcx_settings(self,
                         illumination_geometry: IlluminationGeometryBase,
                         assumed_anisotropy: np.ndarray,
                         **kwargs) -> Dict:
        """
        generates MCX-specific settings for simulations based on Tags in `self.global_settings` and
        `self.component_settings` . Among others, it defines the volume type, dimensions and path to binary file.

        :param illumination_geometry: and instance of `IlluminationGeometryBase` defining the illumination geometry
        :param assumed_anisotropy:
        :param kwargs: dummy, used for class inheritance
        :return: dictionary with settings to be used by MCX
        """
        mcx_volumetric_data_file = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                                   self.global_settings[Tags.VOLUME_NAME] + "_output"
        for name, suffix in self.mcx_output_suffixes.items():
            self.__setattr__(name, mcx_volumetric_data_file + suffix)
            self.temporary_output_files.append(mcx_volumetric_data_file + suffix)
        if Tags.TIME_STEP and Tags.TOTAL_TIME in self.component_settings:
            dt = self.component_settings[Tags.TIME_STEP]
            time = self.component_settings[Tags.TOTAL_TIME]
        else:
            time = 5e-09
            dt = 5e-09
        self.frames = int(time / dt)

        source = illumination_geometry.get_mcx_illuminator_definition(self.global_settings)
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
                        "g": assumed_anisotropy,
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
        return settings_dict

    def get_command(self) -> List:
        """
        generates list of commands to be parse to MCX in a subprocess

        :return: list of MCX commands
        """
        cmd = list()
        cmd.append(self.component_settings[Tags.OPTICAL_MODEL_BINARY_PATH])
        cmd.append("-f")
        cmd.append(self.mcx_json_config_file)
        cmd.append("-O")
        cmd.append("F")
        return cmd

    @staticmethod
    def run_mcx(cmd: List) -> None:
        """
        runs subprocess calling MCX with the flags built with `self.get_command`. Rises a `RuntimeError` if the code
        exit of the subprocess is not 0.

        :param cmd: list defining command to parse to `subprocess.run`
        :return: None
        """
        results = None
        try:
            results = subprocess.run(cmd)
        except:
            raise RuntimeError(f"MCX failed to run: {cmd}, results: {results}")

    def generate_mcx_bin_input(self,
                               absorption_cm: np.ndarray,
                               scattering_cm: np.ndarray,
                               anisotropy: np.ndarray,
                               assumed_anisotropy: np.ndarray) -> None:
        """
        generates binary file containing volume scattering and absorption as input for MCX

        :param absorption_cm: Absorption in units of per centimeter
        :param scattering_cm: Scattering in units of per centimeter
        :param anisotropy: Dimensionless scattering anisotropy
        :param assumed_anisotropy:
        :return: None
        """
        absorption_mm, scattering_mm = self.pre_process_volumes(**{'absorption_cm': absorption_cm,
                                                                   'scattering_cm': scattering_cm,
                                                                   'anisotropy': anisotropy,
                                                                   'assumed_anisotropy': assumed_anisotropy})
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
        self.temporary_output_files.append(tmp_input_path)
        with open(tmp_input_path, "wb") as input_file:
            input_file.write(mcx_input)

        del mcx_input, input_file
        struct._clearcache()
        gc.collect()

    def read_mcx_output(self, **kwargs) -> Dict:
        """
        reads the temporary output generated with MCX

        :param kwargs: dummy, used for class inheritance compatibility
        :return: `Dict` instance containing the MCX output
        """
        with open(self.mcx_volumetric_data_file, 'rb') as f:
            data = f.read()
        data = struct.unpack('%df' % (len(data) / 4), data)
        fluence = np.asarray(data).reshape([self.nx, self.ny, self.nz, self.frames], order='F')
        fluence *= 100  # Convert from J/mm^2 to J/cm^2
        if np.shape(fluence)[3] == 1:
            fluence = np.squeeze(fluence, 3)
        results = dict()
        results[Tags.DATA_FIELD_FLUENCE] = fluence
        return results

    def remove_mcx_output(self) -> None:
        """
        deletes temporary MCX output files from the file system

        :return: None
        """
        for f in self.temporary_output_files:
            if os.path.isfile(f):
                os.remove(f)

    def pre_process_volumes(self, **kwargs) -> Tuple:
        """
        pre-process volumes before running simulations with MCX. The volumes are transformed to `mm` units

        :param kwargs: dictionary containing at least the keys `scattering_cm, absorption_cm, anisotropy` and
            `assumed_anisotropy`
        :return: `Tuple` of volumes after transformation
        """
        return self.volumes_to_mm(**kwargs)

    @staticmethod
    def volumes_to_mm(**kwargs) -> Tuple:
        """
        transforms volumes into `mm` units

        :param kwargs: dictionary containing at least the keys `scattering_cm, absorption_cm, anisotropy` and
            `assumed_anisotropy`
        :return: `Tuple` of volumes after transformation
        """
        scattering_cm = kwargs.get('scattering_cm')
        absorption_cm = kwargs.get('absorption_cm')
        absorption_mm = absorption_cm / 10
        scattering_mm = scattering_cm / 10

        # FIXME Currently, mcx only accepts a single value for the anisotropy.
        #   In order to use the correct reduced scattering coefficient throughout the simulation,
        #   we adjust the scattering parameter to be more accurate in the diffuse regime.
        #   This will lead to errors, especially in the quasi-ballistic regime.

        given_reduced_scattering = (scattering_mm * (1 - kwargs.get('anisotropy')))
        scattering_mm = given_reduced_scattering / (1 - kwargs.get('assumed_anisotropy'))
        scattering_mm[scattering_mm < 1e-10] = 1e-10
        return absorption_mm, scattering_mm

    @staticmethod
    def post_process_volumes(**kwargs) -> Tuple:
        """
        post-processes volumes after MCX simulations. Dummy function implemented for compatibility with inherited
        classes

        :param kwargs: dictionary containing at least the key `volumes` to be transformed
        :return:
        """
        arrays = kwargs.get('arrays')
        return tuple(a for a in arrays)
