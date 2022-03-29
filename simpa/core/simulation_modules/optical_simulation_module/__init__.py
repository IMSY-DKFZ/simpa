# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Union, Dict
from abc import abstractmethod
import gc

from simpa.utils import Tags, Settings
from simpa.core import SimulationModule
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5
from simpa.core.device_digital_twins import IlluminationGeometryBase, PhotoacousticDevice
from simpa.utils.quality_assurance.data_sanity_testing import assert_array_well_defined


class OpticalForwardModuleBase(SimulationModule):
    """
    Use this class as a base for implementations of optical forward models.
    This class has the attributes `self.temporary_output_files` which stores file paths that are temporarily created as
    input to the optical simulator, e.g. MCX. The class attributes `nx, ny & nz` represent the volume dimensions
    """

    def __init__(self, global_settings: Settings):
        super(OpticalForwardModuleBase, self).__init__(global_settings=global_settings)
        self.component_settings = self.global_settings.get_optical_settings()
        self.nx = None
        self.ny = None
        self.nz = None
        self.temporary_output_files = []

    @abstractmethod
    def forward_model(self,
                      absorption_cm: np.ndarray,
                      scattering_cm: np.ndarray,
                      anisotropy: np.ndarray,
                      illumination_geometry: IlluminationGeometryBase):
        """
        A deriving class needs to implement this method according to its model.

        :param absorption_cm: Absorption in units of per centimeter
        :param scattering_cm: Scattering in units of per centimeter
        :param anisotropy: Dimensionless scattering anisotropy
        :param illumination_geometry: A device that represents a detection geometry
        :return: Fluence in units of J/cm^2
        """
        pass

    def run(self, device: Union[IlluminationGeometryBase, PhotoacousticDevice]) -> None:
        """
        runs optical simulations. Volumes are first loaded from HDF5 file and parsed to `self.forward_model`, the output
        is aggregated in case multiple illuminations are defined by `device` and stored in the same HDF5 file.

        :param device: Illumination or Photoacoustic device that defines the illumination geometry
        :return: None
        """

        self.logger.info("Simulating the optical forward process...")

        properties_path = generate_dict_path(Tags.SIMULATION_PROPERTIES,
                                             wavelength=self.global_settings[Tags.WAVELENGTH])

        optical_properties = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH], properties_path)
        absorption = optical_properties[Tags.DATA_FIELD_ABSORPTION_PER_CM][str(self.global_settings[Tags.WAVELENGTH])]
        scattering = optical_properties[Tags.DATA_FIELD_SCATTERING_PER_CM][str(self.global_settings[Tags.WAVELENGTH])]
        anisotropy = optical_properties[Tags.DATA_FIELD_ANISOTROPY][str(self.global_settings[Tags.WAVELENGTH])]
        gruneisen_parameter = optical_properties[Tags.DATA_FIELD_GRUNEISEN_PARAMETER]
        del optical_properties
        gc.collect()

        _device = None
        if isinstance(device, IlluminationGeometryBase):
            _device = device
        elif isinstance(device, PhotoacousticDevice):
            _device = device.get_illumination_geometry()
        else:
            raise TypeError(f"The optical forward modelling does not support devices of type {type(device)}")

        results = self.run_forward_model(_device=_device,
                                         device=device,
                                         absorption=absorption,
                                         scattering=scattering,
                                         anisotropy=anisotropy)
        fluence = results[Tags.DATA_FIELD_FLUENCE]
        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(fluence, assume_non_negativity=True, array_name="fluence")

        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in self.component_settings:
            units = Tags.UNITS_PRESSURE
            # Initial pressure should be given in units of Pascale
            conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
            initial_pressure = (absorption * fluence * gruneisen_parameter *
                                (self.component_settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000)
                                * conversion_factor)
        else:
            units = Tags.UNITS_ARBITRARY
            initial_pressure = absorption * fluence

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(initial_pressure, assume_non_negativity=True, array_name="initial_pressure")

        results[Tags.DATA_FIELD_FLUENCE] = fluence
        results[Tags.OPTICAL_MODEL_UNITS] = units
        results[Tags.DATA_FIELD_INITIAL_PRESSURE] = initial_pressure
        optical_output = {}
        for k, item in results.items():
            optical_output[k] = {self.global_settings[Tags.WAVELENGTH]: item}

        optical_output_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME)
        save_hdf5(optical_output, self.global_settings[Tags.SIMPA_OUTPUT_PATH], optical_output_path)
        self.logger.info("Simulating the optical forward process...[Done]")

    def run_forward_model(self,
                          _device,
                          device: Union[IlluminationGeometryBase, PhotoacousticDevice],
                          absorption: np.ndarray,
                          scattering: np.ndarray,
                          anisotropy: np.ndarray) -> Dict:
        """
        runs `self.forward_model` as many times as defined by `device` and aggregates the results.

        :param _device: device illumination geometry
        :param device: class defining illumination
        :param absorption: Absorption volume
        :param scattering: Scattering volume
        :param anisotropy: Dimensionless scattering anisotropy
        :return:
        """
        if isinstance(_device, list):
            # per convention this list has at least two elements
            results = self.forward_model(absorption_cm=absorption,
                                         scattering_cm=scattering,
                                         anisotropy=anisotropy,
                                         illumination_geometry=_device[0])
            fluence = results[Tags.DATA_FIELD_FLUENCE]
            for idx in range(1, len(_device)):
                # we already looked at the 0th element, so go from 1 to n-1
                results = self.forward_model(absorption_cm=absorption,
                                             scattering_cm=scattering,
                                             anisotropy=anisotropy,
                                             illumination_geometry=_device[idx])
                fluence += results[Tags.DATA_FIELD_FLUENCE]

            fluence = fluence / len(_device)

        else:
            results = self.forward_model(absorption_cm=absorption,
                                         scattering_cm=scattering,
                                         anisotropy=anisotropy,
                                         illumination_geometry=_device)
            fluence = results[Tags.DATA_FIELD_FLUENCE]
        return {Tags.DATA_FIELD_FLUENCE: fluence}
