from simulate import Tags
from io_handling.io_hdf5 import load_hdf5
from abc import abstractmethod


class OpticalForwardAdapterBase:
    """
    Use this class as a base for implementations of optical forward models.
    """

    @abstractmethod
    def forward_model(self, absorption_cm, scattering_cm, anisotropy, settings):
        """
        A deriving class needs to implement this method according to its model.

        :param absorption_cm: Absorption in units of per centimeter
        :param scattering_cm: Scattering in units of per centimeter
        :param anisotropy: Dimensionless scattering anisotropy
        :param settings: Setting dictionary
        :return: Fluence in units of J/cm^2
        """
        pass

    def simulate(self, optical_properties_path, settings):
        """
        Call this method to invoke the simulation process.

        A adapter that implements the forward_model method, will take optical properties of absorption, scattering,
        and scattering anisotropy as input and return the light fluence as output.

        :param optical_properties_path: path to a *.npz file that contains the following tags:
            Tags.PROPERTY_ABSORPTION_PER_CM -> contains the optical absorptions in units of one per centimeter
            Tags.PROPERTY_SCATTERING_PER_CM -> contains the optical scattering in units of one per centimeter
            Tags.PROPERTY_ANISOTROPY -> contains the dimensionless optical scattering anisotropy
        :param settings:
        :return:
        """
        print("Simulating the optical forward process...")

        optical_properties = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], optical_properties_path)
        absorption = optical_properties[Tags.PROPERTY_ABSORPTION_PER_CM]
        scattering = optical_properties[Tags.PROPERTY_SCATTERING_PER_CM]
        anisotropy = optical_properties[Tags.PROPERTY_ANISOTROPY]

        fluence = self.forward_model(absorption_cm=absorption,
                                     scattering_cm=scattering,
                                     anisotropy=anisotropy,
                                     settings=settings)

        print("Simulating the optical forward process...[Done]")
        return fluence