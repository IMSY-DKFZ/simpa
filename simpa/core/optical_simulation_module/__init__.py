"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags, Settings
from abc import abstractmethod
from simpa.core.simulation_components import SimulationModule
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5
import gc


class OpticalForwardModuleBase(SimulationModule):
    """
    Use this class as a base for implementations of optical forward models.
    """

    def __init__(self, global_settings: Settings):
        super(OpticalForwardModuleBase, self).__init__(global_settings=global_settings)
        self.component_settings = self.global_settings.get_optical_settings()

    @abstractmethod
    def forward_model(self, absorption_cm, scattering_cm, anisotropy):
        """
        A deriving class needs to implement this method according to its model.

        :param absorption_cm: Absorption in units of per centimeter
        :param scattering_cm: Scattering in units of per centimeter
        :param anisotropy: Dimensionless scattering anisotropy
        :return: Fluence in units of J/cm^2
        """
        pass

    def run(self):
        """
        Call this method to invoke the simulation process.

        A adapter that implements the forward_model method, will take optical properties of absorption, scattering,
        and scattering anisotropy as input and return the light fluence as output.

        :param optical_properties_path: path to a .npz file that contains the following tags:
            Tags.PROPERTY_ABSORPTION_PER_CM -> contains the optical absorptions in units of one per centimeter
            Tags.PROPERTY_SCATTERING_PER_CM -> contains the optical scattering in units of one per centimeter
            Tags.PROPERTY_ANISOTROPY -> contains the dimensionless optical scattering anisotropy
        :param settings:
        :return:
        """
        self.logger.info("Simulating the optical forward process...")

        properties_path = generate_dict_path(Tags.SIMULATION_PROPERTIES,
                                             wavelength=self.global_settings[Tags.WAVELENGTH])

        optical_properties = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH], properties_path)
        absorption = optical_properties[Tags.PROPERTY_ABSORPTION_PER_CM][str(self.global_settings[Tags.WAVELENGTH])]
        scattering = optical_properties[Tags.PROPERTY_SCATTERING_PER_CM][str(self.global_settings[Tags.WAVELENGTH])]
        anisotropy = optical_properties[Tags.PROPERTY_ANISOTROPY][str(self.global_settings[Tags.WAVELENGTH])]
        gruneisen_parameter = optical_properties[Tags.PROPERTY_GRUNEISEN_PARAMETER]
        del optical_properties
        gc.collect()

        fluence = self.forward_model(absorption_cm=absorption,
                                     scattering_cm=scattering,
                                     anisotropy=anisotropy)

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

        optical_output_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME)

        optical_output = {
            Tags.OPTICAL_MODEL_FLUENCE: {self.global_settings[Tags.WAVELENGTH]: fluence},
            Tags.OPTICAL_MODEL_INITIAL_PRESSURE: {self.global_settings[Tags.WAVELENGTH]: initial_pressure},
            Tags.OPTICAL_MODEL_UNITS: {self.global_settings[Tags.WAVELENGTH]: units}
        }

        save_hdf5(optical_output, self.global_settings[Tags.SIMPA_OUTPUT_PATH], optical_output_path)

        self.logger.info("Simulating the optical forward process...[Done]")
