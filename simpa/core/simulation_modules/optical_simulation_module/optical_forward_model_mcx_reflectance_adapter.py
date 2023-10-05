# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import numpy as np
from collections.abc import Iterable
from typing import Tuple, Dict, Union

from simpa.utils import Tags, Settings
from simpa.core.simulation_modules.optical_simulation_module.optical_forward_model_mcx_adapter import MCXAdapter
from simpa.core.device_digital_twins import IlluminationGeometryBase, PhotoacousticDevice


class MCXAdapterReflectance(MCXAdapter):
    """
    This class implements a bridge to the mcx framework using pmcx to integrate mcx into SIMPA.
    This class implements the capability to run diffuse reflectance simulations.

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
        self.padded = None

    def parse_pmcx_results(self, pmcx_results: Dict) -> Dict:
        results = super(MCXAdapterReflectance, self).parse_pmcx_results(pmcx_results)
        if Tags.COMPUTE_DIFFUSE_REFLECTANCE in self.component_settings and \
                self.component_settings[Tags.COMPUTE_DIFFUSE_REFLECTANCE]:
            pos = np.where(pmcx_results["dref"] > 0)
            ref = pmcx_results["dref"][pos]
            pos = np.array(pos).T  # reformatting to aggregate results after for multi illuminant geometries
            results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE] = ref
            results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS] = pos
        if Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and \
                self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT]:
            results[Tags.DATA_FIELD_PHOTON_EXIT_POS] = pmcx_results["detp"][0:3, :]  # p
            results[Tags.DATA_FIELD_PHOTON_EXIT_DIR] = pmcx_results["detp"][3:6, :]  # v
        return results

    def get_mcx_config(self,
                       illumination_geometry: IlluminationGeometryBase,
                       absorption_cm: np.ndarray,
                       scattering_cm: np.ndarray,
                       anisotropy: np.ndarray,
                       assumed_anisotropy: np.ndarray,
                       **kwargs) -> Dict:
        config = super(MCXAdapterReflectance, self).get_mcx_config(illumination_geometry,
                                                                   absorption_cm, scattering_cm, anisotropy, assumed_anisotropy, **kwargs)
        if Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT in self.component_settings and \
                self.component_settings[Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT]:
            config["issavedet"] = 1
            config["issrcfrom0"] = 1
            config["maxdetphoton"] = int(self.component_settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS])
            config["bc"] = "______000010"  # detect photons exiting in the +y direction
            config["savedetflag"] = "XV"  # save photon exit position and direction
        if Tags.COMPUTE_DIFFUSE_REFLECTANCE in self.component_settings and \
                self.component_settings[Tags.COMPUTE_DIFFUSE_REFLECTANCE]:
            config["issaveref"] = 1  # save diffuse reflectance at 0 filled voxels outside of domain
        return config

    def pre_process_volumes(self, **kwargs) -> Tuple:
        """
        pre-process volumes before running simulations with MCX. The volumes are transformed to `mm` units and pads
        a 0-values layer along the z-axis in order to save diffuse reflectance values. All 0-valued voxels are then
        transformed to `np.nan`. This last transformation `0->np.nan` is a requirement from MCX.

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
        post-processes volumes after MCX simulations.
        Removes layer padded by `self.pre_process_volumes` if it was added and transforms `np.nan -> 0`.

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
            a[np.isnan(a)] = 0
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
        _devices = _device if isinstance(_device, Iterable) else (_device,)
        fluence = None
        for illumination_geometry in _devices:
            results = self.forward_model(absorption_cm=absorption,
                                         scattering_cm=scattering,
                                         anisotropy=anisotropy,
                                         illumination_geometry=illumination_geometry)
            if fluence:
                fluence += results[Tags.DATA_FIELD_FLUENCE]
            else:
                fluence = results[Tags.DATA_FIELD_FLUENCE]
            if Tags.DATA_FIELD_DIFFUSE_REFLECTANCE in results:
                reflectance.append(results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE])
                reflectance_position.append(results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS])
            if Tags.DATA_FIELD_PHOTON_EXIT_POS in results:
                photon_position.append(results[Tags.DATA_FIELD_PHOTON_EXIT_POS])
                photon_direction.append(results[Tags.DATA_FIELD_PHOTON_EXIT_DIR])

        aggregated_results = dict()
        aggregated_results[Tags.DATA_FIELD_FLUENCE] = fluence / len(_devices)
        if reflectance:
            aggregated_results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE] = np.concatenate(reflectance, axis=0)
            aggregated_results[Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS] = np.concatenate(reflectance_position, axis=0)
        if photon_position:
            aggregated_results[Tags.DATA_FIELD_PHOTON_EXIT_POS] = np.concatenate(photon_position, axis=0)
            aggregated_results[Tags.DATA_FIELD_PHOTON_EXIT_DIR] = np.concatenate(photon_direction, axis=0)
        return aggregated_results
