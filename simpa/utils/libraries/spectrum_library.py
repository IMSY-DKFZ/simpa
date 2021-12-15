# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import inspect
import glob
import numpy as np
import matplotlib.pylab as plt
from simpa.utils.libraries.literature_values import OpticalTissueProperties
from simpa.utils.serializer import SerializableSIMPAClass


class Spectrum(SerializableSIMPAClass, object):
    """
    An instance of this class represents the absorption spectrum over wavelength for a particular
    """

    def __init__(self, spectrum_name: str, wavelengths: np.ndarray, values: np.ndarray):
        """
        :param spectrum_name:
        :param wavelengths:
        :param values:
        """
        self.spectrum_name = spectrum_name
        self.wavelengths = wavelengths
        self.max_wavelength = int(np.max(wavelengths))
        self.min_wavelength = int(np.min(wavelengths))
        self.values = values

        if np.shape(wavelengths) != np.shape(values):
            raise ValueError("The shape of the wavelengths and the absorption coefficients did not match: " +
                             str(np.shape(wavelengths)) + " vs " + str(np.shape(values)))

        new_wavelengths = np.arange(self.min_wavelength, self.max_wavelength+1, 1)
        self.new_absorptions = np.interp(new_wavelengths, self.wavelengths, self.values)

    def get_value_over_wavelength(self):
        """
        :return: numpy array with the available wavelengths and the corresponding absorption properties
        """
        return np.asarray([self.wavelengths, self.values])

    def get_value_for_wavelength(self, wavelength: int) -> float:
        """
        :param wavelength: the wavelength to retrieve a optical absorption value for [cm^{-1}].
                           Must be an integer value between the minimum and maximum wavelength.
        :return: the best matching linearly interpolated absorption value for the given wavelength.
        """
        return self.new_absorptions[wavelength-self.min_wavelength]

    def __eq__(self, other):
        if isinstance(other, Spectrum):
            return (self.spectrum_name == other.spectrum_name,
                    self.wavelengths == other.wavelengths,
                    self.values == other.values)
        else:
            return super().__eq__(other)

    def serialize(self) -> dict:
        serialized_spectrum = self.__dict__
        return {"Spectrum": serialized_spectrum}

    @staticmethod
    def deserialize(dictionary_to_deserialize: dict):
        deserialized_spectrum = Spectrum(spectrum_name=dictionary_to_deserialize["spectrum_name"],
                                         wavelengths=dictionary_to_deserialize["wavelengths"],
                                         values=dictionary_to_deserialize["values"])
        return deserialized_spectrum


class SpectraLibrary(object):

    def __init__(self, folder_name: str, additional_folder_path: str = None):
        self.spectra = list()
        self.add_spectra_from_folder(folder_name)
        if additional_folder_path is not None:
            self.add_spectra_from_folder(additional_folder_path)

    def add_spectra_from_folder(self, folder_name):
        base_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        for absorption_spectrum in glob.glob(os.path.join(base_path, folder_name, "*.npz")):
            name = absorption_spectrum.split(os.path.sep)[-1][:-4]
            numpy_data = np.load(absorption_spectrum)
            values = numpy_data["values"]
            wavelengths = numpy_data["wavelengths"]
            self.spectra.append(Spectrum(spectrum_name=name, values=values, wavelengths=wavelengths))

    def __next__(self):
        if self.i > 0:
            self.i -= 1
            return self.spectra[self.i]
        raise StopIteration()

    def __iter__(self):
        self.i = len(self.spectra)
        return self

    def get_spectra_names(self):
        return [spectrum.spectrum_name for spectrum in self]

    def get_spectrum_by_name(self, spectrum_name: str) -> Spectrum:
        for spectrum in self:
            if spectrum.spectrum_name == spectrum_name:
                return spectrum

        raise LookupError(f"No spectrum for the given name exists ({spectrum_name}). Try one of: {self.get_spectra_names()}")


class AnisotropySpectrumLibrary(SpectraLibrary):

    def __init__(self, additional_folder_path: str = None):
        super(AnisotropySpectrumLibrary, self).__init__("anisotropy_spectra_data", additional_folder_path)

    @staticmethod
    def CONSTANT_ANISOTROPY_ARBITRARY(anisotropy: float = 1):
        return Spectrum("Constant Anisotropy (arb)", np.asarray([450, 1000]),
                        np.asarray([anisotropy, anisotropy]))


class ScatteringSpectrumLibrary(SpectraLibrary):

    def __init__(self, additional_folder_path: str = None):
        super(ScatteringSpectrumLibrary, self).__init__("scattering_spectra_data", additional_folder_path)

    @staticmethod
    def CONSTANT_SCATTERING_ARBITRARY(scattering: float = 1):
        return Spectrum("Constant Scattering (arb)", np.asarray([450, 1000]),
                        np.asarray([scattering, scattering]))

    @staticmethod
    def scattering_from_rayleigh_and_mie_theory(name: str, mus_at_500_nm: float = 1.0, fraction_rayleigh_scattering: float = 0.0,
                                                mie_power_law_coefficient: float = 0.0):
        wavelengths = np.arange(450, 1001, 1)
        scattering = (mus_at_500_nm * (fraction_rayleigh_scattering * (wavelengths / 500) ** 1e-4 +
                      (1 - fraction_rayleigh_scattering) * (wavelengths / 500) ** -mie_power_law_coefficient))
        return Spectrum(name, wavelengths, scattering)


class AbsorptionSpectrumLibrary(SpectraLibrary):

    def __init__(self, additional_folder_path: str = None):
        super(AbsorptionSpectrumLibrary, self).__init__("absorption_spectra_data", additional_folder_path)

    @staticmethod
    def CONSTANT_ABSORBER_ARBITRARY(absorption_coefficient: float = 1):
        return Spectrum("Constant Absorber (arb)", np.asarray([450, 1000]),
                        np.asarray([absorption_coefficient, absorption_coefficient]))


def get_simpa_internal_absorption_spectra_by_names(absorption_spectrum_names: list):
    lib = AbsorptionSpectrumLibrary()
    spectra = []
    for spectrum_name in absorption_spectrum_names:
        spectra.append(lib.get_spectrum_by_name(spectrum_name))
    return spectra


def view_saved_spectra(save_path=None, mode="absorption"):
    """
    Opens a matplotlib plot and visualizes the available absorption spectra.

    :param save_path: If not None, then the figure will be saved as a png file to the destination.
    :param mode: string that is "absorption", "scattering", or "anisotropy"
    """
    plt.figure(figsize=(11, 8))
    if mode == "absorption":
        for spectrum in AbsorptionSpectrumLibrary():
            plt.semilogy(spectrum.wavelengths,
                         spectrum.values,
                         label=spectrum.spectrum_name)
    if mode == "scattering":
        for spectrum in ScatteringSpectrumLibrary():
            plt.semilogy(spectrum.wavelengths,
                         spectrum.values,
                         label=spectrum.spectrum_name)
    if mode == "anisotropy":
        for spectrum in AnisotropySpectrumLibrary():
            plt.semilogy(spectrum.wavelengths,
                         spectrum.values,
                         label=spectrum.spectrum_name)
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_ylabel(mode)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_title(f"{mode} spectra for all absorbers present in the library")
    # ax.hlines([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 450, 1000, linestyles="dashed", colors=["#EEEEEE88"])
    ax.legend(loc='best', bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        plt.savefig(save_path + f"{mode}_spectra.png")
    plt.show()
    plt.close()
