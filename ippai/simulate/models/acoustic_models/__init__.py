from ippai.simulate import Tags
import numpy as np
from abc import abstractmethod

class AcousticForwardAdapterBase:
    """
    Use this class as a base for implementations of optical forward models.
    """

    @abstractmethod
    def forward_model(self, settings):
        """
        A deriving class needs to implement this method according to its model.

        :param settings: Setting dictionary
        :return: Fluence in units of J/cm^2
        """
        pass

    def simulate(self, optical_properties_path, settings):
        """
        Call this method to invoke the simulation process.
        TODO

        A adapter that implements the forward_model method, will take acoustic properties as input
        and return the time series pressure data as output.

        :param optical_properties_path: path to a *.npz file that contains the following tags:
            Tags.PROPERTY_ABSORPTION_PER_CM -> contains the optical absorptions in units of one per centimeter
            Tags.PROPERTY_SCATTERING_PER_CM -> contains the optical scattering in units of one per centimeter
            Tags.PROPERTY_ANISOTROPY -> contains the dimensionless optical scattering anisotropy
        :param settings:
        :return:
        """
        print("Simulating the acoustic forward process...")

        time_series_pressure_data = self.forward_model(settings=settings)

        print("Simulating the acoustic forward process...[Done]")
        return time_series_pressure_data