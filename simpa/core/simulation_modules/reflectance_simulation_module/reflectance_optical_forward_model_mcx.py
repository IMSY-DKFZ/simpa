"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import numpy as np
import struct
import subprocess
from simpa.utils import Tags
from simpa.core.simulation_modules.optical_simulation_module import OpticalForwardModuleBase
import json
import os
import gc


class ReflectanceOpticalForwardModelMcxAdapter(OpticalForwardModuleBase):
    """
    This class implements a bridge to the mcx framework to integrate mcx into SIMPA.
    MCX is a GPU-enabled Monte-Carlo model simulation of photon transport in tissue::

        Fang, Qianqian, and David A. Boas. "Monte Carlo simulation of photon migration in 3D
        turbid media accelerated by graphics processing units."
        Optics express 17.22 (2009): 20178-20190.

    """

    def forward_model(self,
                      absorption_cm,
                      scattering_cm,
                      anisotropy,
                      illumination_geometry,
                      probe_position_mm,
                      saveexit: bool = False,
                      saveref: bool = True):

        if Tags.MCX_ASSUMED_ANISOTROPY in self.component_settings:
            _assumed_anisotropy = self.component_settings[Tags.MCX_ASSUMED_ANISOTROPY]
        else:
            _assumed_anisotropy = 0.9

        absorption_mm = absorption_cm / 10
        scattering_mm = scattering_cm / 10

        # FIXME Currently, mcx only accepts a single value for the anisotropy.
        #   In order to use the correct reduced scattering coefficient throughout the simulation,
        #   we adjust the scattering parameter to be more accurate in the diffuse regime.
        #   This will lead to errors, especially in the quasi-ballistic regime.

        given_reduced_scattering = (scattering_mm * (1 - anisotropy))
        scattering_mm = given_reduced_scattering / (1 - _assumed_anisotropy)
        scattering_mm[scattering_mm < 1e-10] = 1e-10

        op_array = np.asarray([absorption_mm, scattering_mm])

        [_, nx, ny, nz] = np.shape(op_array)

        # create a binary of the volume

        optical_properties_list = list(np.reshape(op_array, op_array.size, "F"))
        del absorption_cm, absorption_mm, scattering_cm, scattering_mm, op_array
        gc.collect()
        mcx_input = struct.pack("f" * len(optical_properties_list), *optical_properties_list)
        del optical_properties_list
        gc.collect()
        tmp_input_path = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                         self.global_settings[Tags.VOLUME_NAME]+".bin"
        with open(tmp_input_path, "wb") as input_file:
            input_file.write(mcx_input)

        del mcx_input, input_file
        struct._clearcache()
        gc.collect()

        tmp_output_file = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                          self.global_settings[Tags.VOLUME_NAME]+"_output"

        # write settings to json
        # time = 1.16e-09
        # dt = 8e-12
        if Tags.TIME_STEP and Tags.TOTAL_TIME in self.component_settings:
            dt = self.component_settings[Tags.TIME_STEP]
            time = self.component_settings[Tags.TOTAL_TIME]
        else:
            time = 5e-09
            dt = 5e-09
        frames = int(time/dt)

        source = illumination_geometry.get_mcx_illuminator_definition(self.global_settings, probe_position_mm)

        settings_dict = {
            "Session": {
                "ID": tmp_output_file,
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
                "Dim": [nx, ny, nz],
                "VolumeFile": self.global_settings[Tags.SIMULATION_PATH] + "/" +
                              self.global_settings[Tags.VOLUME_NAME]+".bin"
            }}

        if Tags.MCX_SEED not in self.component_settings:
            if Tags.RANDOM_SEED in self.global_settings:
                settings_dict["Session"]["RNGSeed"] = self.global_settings[Tags.RANDOM_SEED]
        else:
            settings_dict["Session"]["RNGSeed"] = self.component_settings[Tags.MCX_SEED]

        print(settings_dict)

        tmp_json_filename = self.global_settings[Tags.SIMULATION_PATH] + "/" + \
                            self.global_settings[Tags.VOLUME_NAME]+".json"
        with open(tmp_json_filename, "w") as json_file:
            json.dump(settings_dict, json_file, indent="\t")

        # run the simulation

        cmd = list()
        cmd.append(self.component_settings[Tags.OPTICAL_MODEL_BINARY_PATH])
        cmd.append("-f")
        cmd.append(tmp_json_filename)
        cmd.append("-O")
        cmd.append("F")
        if saveexit:
            cmd.append("-x")  # save photon exit position and direction
            cmd.append("1")
        if saveref:
            cmd.append("-X")  # save diffuse reflectance at 0 filled voxels outside of domain
            cmd.append("1")
        res = subprocess.run(cmd)

        # Read output

        with open(tmp_output_file+".mc2", 'rb') as f:
            data = f.read()
        data = struct.unpack('%df' % (len(data) / 4), data)
        fluence = np.asarray(data).reshape([nx, ny, nz, frames], order='F')
        if np.shape(fluence)[3] == 1:
            fluence = np.squeeze(fluence, 3) * 100  # Convert from J/mm^2 to J/cm^2

        struct._clearcache()

        os.remove(tmp_input_path)
        os.remove(tmp_output_file+".mc2")
        os.remove(tmp_json_filename)

        return fluence

    def get_surface_from_volume(volume: np.ndarray, ax: int = 2) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Locates the position at which diffuse reflectance is stored in volume. That is the first layer along last axis along
        which all values are different than 0. a surface is defined as the collection of values along an axis with values
        different than 0. If all values along specified axis are 0 for one location, the position of surface at that
        location is set to `volume.shape[ax] - 1`.

        :param volume: array from which the surface wants to be obtained along a given axis
        :param ax: default=2, axis along which surface is desired
        :return: tuple containing the position of the surface along specified axis, can be used to slice volume
        """
        volume_non_zero = volume != 0
        pos = np.argmax(volume_non_zero, axis=ax)
        #TODO: apply_along_axis is too slow, needs to be replaced
        all_zero_lines = np.where(np.apply_along_axis(np.all, ax, volume == 0))
        pos[all_zero_lines] = volume.shape[ax] - 1
        pos = np.where(np.isreal(pos)) + (pos.flatten(),)
        if ax == 0:
            pos = (pos[2], pos[0], pos[1])
        elif ax == 1:
            pos = (pos[0], pos[2], pos[1])
        return pos

    def check_volumes(self, volumes: dict) -> None:
        """
        Checks consistency of volumes. Helpful when parsing custom volumes to simulations. Checks that all volumes have same
        shape, that the minimum required volumes are defined and that values are meaningful for specific volumes.
        For example: `np.all(volumes['mua'] >= 0)`. It also checks that there are no nan or inf values in all volumes.
        :param volumes: dictionary containing the volumes to check
        :return: None
        """
        shapes = []
        required_vol_keys = Tags.MINIMUM_REQUIRED_VOLUMES
        for key in volumes:
            vol = volumes[key]
            if np.any(np.isnan(vol)) or np.any(np.isinf(vol)):
                raise ValueError(f"Found nan or inf value in volume: {key}")
            if key in ["mua", "mus"] and not np.all(vol >= 0):
                raise ValueError(f"Found negative value in volume: {key}")
            shapes.append(vol.shape)
        g = groupby(shapes)
        shapes_equal = next(g, True) and not next(g, False)
        if not shapes_equal:
            raise ValueError(f"Not all shapes of custom volumes are equal: {shapes}")
        for key in required_vol_keys:
            if key not in volumes.keys():
                raise ValueError(f"Could not find required key in custom volumes {key}")

    def extract_diffuse_reflectance(self, volumes: dict) -> dict:
        """
        Extracts the diffuse reflectance layer from fluence volume and stores in a new key inside volumes. Then sets all
        values in volumes where the diffuse reflectance is located to 0.
        :param volumes: dictionary with original volumes
        :return: dictionary containing the corrected volumes and the diffuse reflectance
        """
        fluence = volumes[Tags.OPTICAL_MODEL_FLUENCE]
        dr_layer_pos = calculate.get_surface_from_volume(fluence)
        diffuse_reflectance = fluence[dr_layer_pos]
        diffuse_reflectance = diffuse_reflectance.reshape(fluence.shape[:2])
        for key in volumes:
            volume = volumes[key]
            volume[dr_layer_pos] -= volume[dr_layer_pos]
        volumes[Tags.OPTICAL_MODEL_DIFFUSE_REFLECTANCE] = diffuse_reflectance
        volumes[Tags.SURFACE_LAYER_POSITION] = dr_layer_pos
        return volumes