"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags, Settings
from abc import abstractmethod
from simpa.core import SimulationModule
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5
import gc
from simpa.core.device_digital_twins import IlluminationGeometryBase, PhotoacousticDevice
from simpa.utils.quality_assurance.data_sanity_testing import assert_array_well_defined


class OpticalForwardModuleBase(SimulationModule):
    """
    Use this class as a base for implementations of optical forward models.
    """

    def __init__(self, global_settings: Settings):
        super(OpticalForwardModuleBase, self).__init__(global_settings=global_settings)
        self.component_settings = self.global_settings.get_optical_settings()

    @abstractmethod
    def forward_model(self, absorption_cm, scattering_cm, anisotropy, illumination_geometry, probe_position_mm):
        """
        A deriving class needs to implement this method according to its model.

        :param absorption_cm: Absorption in units of per centimeter
        :param scattering_cm: Scattering in units of per centimeter
        :param anisotropy: Dimensionless scattering anisotropy
        :param illumination_geometry: A device that represents a detection geometry
        :return: Fluence in units of J/cm^2
        """
        pass

    def run(self, device):

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

        if isinstance(_device, list):
            # per convention this list has at least two elements
            fluence = self.forward_model(absorption_cm=absorption,
                                         scattering_cm=scattering,
                                         anisotropy=anisotropy,
                                         illumination_geometry=_device[0],
                                         probe_position_mm=device.device_position_mm)
            for idx in range(len(_device)-1):
                # we already looked at the 0th element, so go from 1 to n-1
                fluence += self.forward_model(absorption_cm=absorption,
                                              scattering_cm=scattering,
                                              anisotropy=anisotropy,
                                              illumination_geometry=_device[idx+1],
                                              probe_position_mm=device.device_position_mm)

            fluence = fluence / len(_device)

        else:
            fluence = self.forward_model(absorption_cm=absorption,
                                         scattering_cm=scattering,
                                         anisotropy=anisotropy,
                                         illumination_geometry=_device,
                                         probe_position_mm=device.device_position_mm)

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(fluence, assume_non_negativity=True)

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
            assert_array_well_defined(initial_pressure, assume_non_negativity=True)

        optical_output_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME)

        optical_output = {
            Tags.DATA_FIELD_FLUENCE: {self.global_settings[Tags.WAVELENGTH]: fluence},
            Tags.DATA_FIELD_INITIAL_PRESSURE: {self.global_settings[Tags.WAVELENGTH]: initial_pressure},
            Tags.OPTICAL_MODEL_UNITS: {self.global_settings[Tags.WAVELENGTH]: units}
        }

        save_hdf5(optical_output, self.global_settings[Tags.SIMPA_OUTPUT_PATH], optical_output_path)

        self.logger.info("Simulating the optical forward process...[Done]")
