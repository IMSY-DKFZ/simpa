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

from simpa.utils import Tags
from simpa.io_handling.io_hdf5 import load_hdf5
from abc import abstractmethod
from simpa.core import SimulationComponent
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5


class OpticalForwardComponentBase(SimulationComponent):
    """
    Use this class as a base for implementations of optical forward models.
    """

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

        fluence = self.forward_model(absorption_cm=absorption,
                                     scattering_cm=scattering,
                                     anisotropy=anisotropy)

        optical_properties = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH], properties_path)
        absorption = optical_properties[Tags.PROPERTY_ABSORPTION_PER_CM][str(self.global_settings[Tags.WAVELENGTH])]
        gruneisen_parameter = optical_properties[Tags.PROPERTY_GRUNEISEN_PARAMETER]

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
