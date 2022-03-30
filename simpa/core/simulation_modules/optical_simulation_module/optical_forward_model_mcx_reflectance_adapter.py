"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
import numpy as np
import struct
import jdata
import os
from typing import List, Tuple, Dict, Union

from simpa.utils import Tags, Settings
from simpa.core.simulation_modules.optical_simulation_module.optical_forward_model_mcx_adapter import MCXAdapter
from simpa.core.device_digital_twins import IlluminationGeometryBase, PhotoacousticDevice


class MCXAdapterReflectance(MCXAdapter):
    """
    This class implements a bridge to the mcx framework to integrate mcx into SIMPA. This class targets specifically
    diffuse reflectance simulations. Specifically, it implements the capability to run diffuse reflectance simulations.

    .. warning::
        This MCX adapter requires a version of MCX containing the revision: `Rev::077060`, which was published in the
        Nightly build  on `2022-01-26`.

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
        super(MCXAdapterReflectance, self).__init__(global_settings=global_settings)
        self.mcx_photon_data_file = None
        self.padded = None
        self.mcx_output_suffixes = {'mcx_volumetric_data_file': '.jnii',
                                    'mcx_photon_data_file': '_detp.jdat'}

    def forward_model(self,
                      absorption_cm: np.ndarray,
                      scattering_cm: np.ndarray,
                      anisotropy: np.ndarray,
                      illumination_geometry: IlluminationGeometryBase) -> Dict:
        """
        runs the MCX simulations. Binary file containing scattering and absorption volumes is temporarily created as
        input for MCX. A JSON serializable file containing the configuration required by MCx is also generated.
        The set of flags parsed to MCX is built based on the Tags declared in `self.component_settings`, the results
        from MCX are used to populate an instance of Settings and returned.

        :param absorption_cm: array containing the absorption of the tissue in `cm` units
        :param scattering_cm: array containing the scattering of the tissue in `cm` units
        :param anisotropy: array containing the anisotropy of the volume defined by `absorption_cm` and `scattering_cm`
        :param illumination_geometry: and instance of `IlluminationGeometryBase` defining the illumination geometry
        :param probe_position_mm: position of a probe in `mm` units. This is parsed to
            `illumination_geometry.get_mcx_illuminator_definition`
        :return: `Settings` containing the results of optical simulations, the keys in this dictionary-like object
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
                                              assumed_anisotropy=_assumed_anisotropy,
                                              )

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
        cmd.append("-F")
        cmd.append("jnii")
        if Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and \
                self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT]:
            cmd.append("-H")
            cmd.append(f"{int(self.component_settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS])}")
            cmd.append("--bc")  # save photon exit position and direction
            cmd.append("______000010")
            cmd.append("--savedetflag")
            cmd.append("XV")
        if Tags.COMPUTE_DIFFUSE_REFLECTANCE in self.component_settings and \
                self.component_settings[Tags.COMPUTE_DIFFUSE_REFLECTANCE]:
            cmd.append("--saveref")  # save diffuse reflectance at 0 filled voxels outside of domain
        return cmd

    def read_mcx_output(self, **kwargs) -> Dict:
        """
        reads the temporary output generated with MCX

        :param kwargs: dummy, used for class inheritance compatibility
        :return: `Settings` instance containing the MCX output
        """
        results = dict()
        if os.path.isfile(self.mcx_volumetric_data_file) and self.mcx_volumetric_data_file.endswith(
                self.mcx_output_suffixes['mcx_volumetric_data_file']):
            content = jdata.load(self.mcx_volumetric_data_file)
            fluence = content['NIFTIData']
            ref, ref_pos, fluence = self.extract_reflectance_from_fluence(fluence=fluence)
            fluence = self.post_process_volumes(**{'arrays': (fluence,)})[0]
            fluence *= 100  # Convert from J/mm^2 to J/cm^2
            results[Tags.DATA_FIELD_FLUENCE] = fluence
        else:
            raise FileNotFoundError(f"Could not find .jnii file for {self.mcx_volumetric_data_file}")
        if Tags.COMPUTE_DIFFUSE_REFLECTANCE in self.component_settings and \
                self.component_settings[Tags.COMPUTE_DIFFUSE_REFLECTANCE]:
            results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE] = ref
            results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS] = ref_pos
        if Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and \
                self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT]:
            content = jdata.load(self.mcx_photon_data_file)
            photon_pos = content['MCXData']['PhotonData']['p']
            photon_dir = content['MCXData']['PhotonData']['v']
            results[Tags.DATA_FIELD_PHOTON_EXIT_POS] = photon_pos
            results[Tags.DATA_FIELD_PHOTON_EXIT_DIR] = photon_dir
        return results

    @staticmethod
    def extract_reflectance_from_fluence(fluence: np.ndarray) -> Tuple:
        """
        extracts diffuse reflectance from volumes. MCX stores diffuse reflectance as negative values in the fluence
        volume. The position where the reflectance is stored is also returned. If there are no negative values in the
        fluence, `None` is returned instead of reflectance and reflectance position. Negative values in fluence are
        set to `0` after extraction of the reflectance.

        :param fluence: array containing fluence as generated by MCX
        :return: tuple of reflectance, reflectance position and transformed fluence
        """
        if np.any(fluence < 0):
            pos = np.where(fluence < 0)
            ref = fluence[pos] * -1
            pos = np.array(pos).T  # reformatting to aggregate results after for multi illuminant geometries
            fluence[fluence < 0] = 0
            return ref, pos, fluence
        else:
            return None, None, fluence

    def pre_process_volumes(self, **kwargs) -> Tuple:
        """
        pre-process volumes before running simulations with MCX. The volumes are transformed to `mm` units and pads
        a 0-values layer along the z-axis in order to save diffuse reflectance values. All 0-valued voxels are then
        transformed to `np.nan`. This last transformation `0->np.nan` is a requirement form MCX.

        :param kwargs: dictionary containing at least the keys `scattering_cm, absorption_cm, anisotropy` and
            `assumed_anisotropy`
        :return: `Tuple` of volumes after transformation
        """
        arrays = self.volumes_to_mm(**kwargs)
        assert np.all([len(a.shape) == 3] for a in arrays)
        check_padding = (Tags.COMPUTE_DIFFUSE_REFLECTANCE in self.component_settings and
                         self.component_settings[Tags.COMPUTE_DIFFUSE_REFLECTANCE]) or \
                        (Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and
                         self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT])
        # check that all volumes on first layer along z have only 0 values
        if np.any([np.any(a[:, :, 0] != 0)] for a in arrays) and check_padding:
            results = tuple(np.pad(a, ((0, 0), (0, 0), (1, 0)), "constant", constant_values=0) for a in arrays)
            self.padded = True
        else:
            results = tuple(arrays)
            self.padded = False
        for a in results:
            # MCX requires NAN values to store photon direction and reflectance when using float mus and mua
            a[a == 0] = np.nan
        return results

    def post_process_volumes(self, **kwargs):
        """
        post-processes volumes after MCX simulations. Dummy function implemented for compatibility with inherited
        classes. Removes layer padded by `self.pre_process_volumes` if it was added and transforms `np.nan -> 0`.

        :param kwargs: dictionary containing at least the key `volumes` to be transformed
        :return:
        """
        arrays = kwargs.get('arrays')
        if self.padded:
            results = tuple(a[..., 1:] for a in arrays)
        else:
            results = tuple(arrays)
        for a in results:
            # revert nan transformation that was done while pre-processing volumes
            a[np.isnan(a)] = 0.
        return results

    def run_forward_model(self,
                          _device,
                          device: Union[IlluminationGeometryBase, PhotoacousticDevice],
                          absorption: np.ndarray,
                          scattering: np.ndarray,
                          anisotropy: np.ndarray
                          ) -> Dict:
        """
        runs `self.forward_model` as many times as defined by `device` and aggregates the results.

        :param _device: device illumination geometry
        :param device: class defining illumination
        :param absorption: Absorption volume
        :param scattering: Scattering volume
        :param anisotropy: Dimensionless scattering anisotropy
        :return:
        """
        reflectance = []
        reflectance_position = []
        photon_position = []
        photon_direction = []
        if isinstance(_device, list):
            # per convention this list has at least two elements
            results = self.forward_model(absorption_cm=absorption,
                                         scattering_cm=scattering,
                                         anisotropy=anisotropy,
                                         illumination_geometry=_device[0])
            self._append_results(results=results,
                                 reflectance=reflectance,
                                 reflectance_position=reflectance_position,
                                 photon_position=photon_position,
                                 photon_direction=photon_direction)
            fluence = results[Tags.DATA_FIELD_FLUENCE]
            for idx in range(1, len(_device)):
                # we already looked at the 0th element, so go from 1 to n-1
                results = self.forward_model(absorption_cm=absorption,
                                             scattering_cm=scattering,
                                             anisotropy=anisotropy,
                                             illumination_geometry=_device[idx])
                self._append_results(results=results,
                                     reflectance=reflectance,
                                     reflectance_position=reflectance_position,
                                     photon_position=photon_position,
                                     photon_direction=photon_direction)
                fluence += results[Tags.DATA_FIELD_FLUENCE]

            fluence = fluence / len(_device)

        else:
            results = self.forward_model(absorption_cm=absorption,
                                         scattering_cm=scattering,
                                         anisotropy=anisotropy,
                                         illumination_geometry=_device)
            self._append_results(results=results,
                                 reflectance=reflectance,
                                 reflectance_position=reflectance_position,
                                 photon_position=photon_position,
                                 photon_direction=photon_direction)
            fluence = results[Tags.DATA_FIELD_FLUENCE]
        aggregated_results = dict()
        aggregated_results[Tags.DATA_FIELD_FLUENCE] = fluence
        if reflectance:
            aggregated_results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE] = np.concatenate(reflectance, axis=0)
            aggregated_results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS] = np.concatenate(reflectance_position, axis=0)
        if photon_position:
            aggregated_results[Tags.DATA_FIELD_PHOTON_EXIT_POS] = np.concatenate(photon_position, axis=0)
            aggregated_results[Tags.DATA_FIELD_PHOTON_EXIT_DIR] = np.concatenate(photon_direction, axis=0)
        return aggregated_results

    @staticmethod
    def _append_results(results,
                        reflectance,
                        reflectance_position,
                        photon_position,
                        photon_direction):
        if Tags.DATA_FIELD_DIFFUSE_REFLECTANCE in results:
            reflectance.append(results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE])
            reflectance_position.append(results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS])
        if Tags.DATA_FIELD_PHOTON_EXIT_POS in results:
            photon_position.append(results[Tags.DATA_FIELD_PHOTON_EXIT_POS])
            photon_direction.append(results[Tags.DATA_FIELD_PHOTON_EXIT_DIR])
