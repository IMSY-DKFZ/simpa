# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import inspect
import glob
import numpy as np
import matplotlib.pylab as plt
import torch
from scipy import interpolate
from simpa.utils.serializer import SerializableSIMPAClass


class Spectrum(SerializableSIMPAClass, object):
    """
    An instance of this class represents a spectrum over a range of wavelengths.

    Attributes:
        spectrum_name (str): The name of the spectrum.
        wavelengths (np.ndarray): Array of wavelengths.
        max_wavelength (int): Maximum wavelength in the spectrum.
        min_wavelength (int): Minimum wavelength in the spectrum.
        values (np.ndarray): Corresponding values for each wavelength.
        values_interp (np.ndarray): Interpolated values across a continuous range of wavelengths.
    """

    def __init__(self, spectrum_name: str, wavelengths: np.ndarray, values: np.ndarray):
        """
        Initializes a Spectrum instance.

        :param spectrum_name: Name of the spectrum.
        :param wavelengths: Array of wavelengths.
        :param values: Corresponding values of the spectrum at each wavelength.

        :raises ValueError: If the shape of wavelengths does not match the shape of values.
        """
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        wavelengths = torch.from_numpy(wavelengths)
        self.spectrum_name = spectrum_name
        self.wavelengths = wavelengths
        self.max_wavelength = int(torch.floor(torch.max(wavelengths)))
        self.min_wavelength = int(torch.ceil(torch.min(wavelengths)))
        self.values = values

        if torch.Tensor.size(wavelengths) != torch.Tensor.size(values):
            raise ValueError("The shape of the wavelengths and the values did not match: " +
                             str(torch.Tensor.size(wavelengths)) + " vs " + str(torch.Tensor.size(values)))

        new_wavelengths = torch.arange(self.min_wavelength, self.max_wavelength+1, 1)
        new_absorptions_function = interpolate.interp1d(self.wavelengths, self.values)
        self.values_interp = new_absorptions_function(new_wavelengths)

    def get_value_over_wavelength(self) -> np.ndarray:
        """
        Returns an array with the available wavelengths and their corresponding values.

        :return: numpy array with the available wavelengths and the corresponding properties
        """
        return np.asarray([self.wavelengths, self.values])

    def get_value_for_wavelength(self, wavelength: int) -> float:
        """
        Retrieves the interpolated value for a given wavelength within the spectrum range.


        :param wavelength: the wavelength to retrieve a value from the defined spectrum.
                           Must be an integer value between the minimum and maximum wavelength.
        :return: the best matching linearly interpolated values for the given wavelength.
        :raises ValueError: if the given wavelength is not within the range of the spectrum.
        """
        if wavelength < self.min_wavelength or wavelength > self.max_wavelength:
            raise ValueError(f"The given wavelength ({wavelength}) is not within the range of the spectrum "
                             f"({self.min_wavelength} - {self.max_wavelength})")
        return self.values_interp[wavelength-self.min_wavelength]

    def __eq__(self, other):
        """
        Compares two Spectrum objects for equality.

        :param other: Another Spectrum object to compare with.

        :return: True if both objects are equal, False otherwise.
        """
        if isinstance(other, Spectrum):
            return (self.spectrum_name == other.spectrum_name,
                    self.wavelengths == other.wavelengths,
                    self.values == other.values)
        else:
            return super().__eq__(other)

    def serialize(self) -> dict:
        """
        Serializes the spectrum instance into a dictionary format.

        :return: Dictionary representation of the Spectrum instance.
        """
        serialized_spectrum = self.__dict__
        return {"Spectrum": serialized_spectrum}

    @staticmethod
    def deserialize(dictionary_to_deserialize: dict):
        """
        Static method to deserialize a dictionary representation back into a Spectrum object.

        :param dictionary_to_deserialize: Dictionary containing the serialized Spectrum object.

        :return: Deserialized Spectrum object.
        """
        deserialized_spectrum = Spectrum(spectrum_name=dictionary_to_deserialize["spectrum_name"],
                                         wavelengths=dictionary_to_deserialize["wavelengths"],
                                         values=dictionary_to_deserialize["values"])
        return deserialized_spectrum


class SpectraLibrary(object):
    """
    A library to manage and store spectral data.

    This class provides functionality to load and manage spectra data from specified folders.

    Attributes:
        spectra (list): A list to store spectra objects.
    """

    def __init__(self, folder_name: str, additional_folder_path: str = None):
        """
        Initializes the SpectraLibrary with spectra data from the specified folder(s).

        :param folder_name: The name of the folder containing spectra data files.
        :param additional_folder_path: An additional folder path for more spectra data.
        """
        self.spectra = list()
        self.add_spectra_from_folder(folder_name)
        if additional_folder_path is not None:
            self.add_spectra_from_folder(additional_folder_path)

    def add_spectra_from_folder(self, folder_name: str):
        """
        Adds spectra from a specified folder to the spectra list.

        :param folder_name: The name of the folder containing spectra data files.
        """
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

    def get_spectra_names(self) -> list:
        """
        Returns the names of all spectra in the library.

        :return: List of spectra names.
        """
        return [spectrum.spectrum_name for spectrum in self]

    def get_spectrum_by_name(self, spectrum_name: str) -> Spectrum:
        """
        Retrieves a spectrum by its name.

        :param spectrum_name: The name of the spectrum to retrieve.
        :return: The spectrum with the specified name.
        :raises LookupError: If no spectrum with the given name exists.
        """
        for spectrum in self:
            if spectrum.spectrum_name == spectrum_name:
                return spectrum

        raise LookupError(
            f"No spectrum for the given name exists ({spectrum_name}). Try one of: {self.get_spectra_names()}")


class AnisotropySpectrumLibrary(SpectraLibrary):
    """
    A library to manage and store anisotropy spectra data.
    """

    def __init__(self, additional_folder_path: str = None):
        """
        Initializes the AnisotropySpectrumLibrary with anisotropy spectra data.

        :param additional_folder_path: An additional folder path for more anisotropy spectra data.
        """
        super(AnisotropySpectrumLibrary, self).__init__("anisotropy_spectra_data", additional_folder_path)

    @staticmethod
    def CONSTANT_ANISOTROPY_ARBITRARY(anisotropy: float = 1) -> Spectrum:
        """
        Creates a constant anisotropy spectrum with an arbitrary value.

        :param anisotropy: The anisotropy value to use.
        :return: A Spectrum instance with constant anisotropy.
        """
        return Spectrum("Constant Anisotropy (arb)", np.asarray([450, 1000]),
                        np.asarray([anisotropy, anisotropy]))


class ScatteringSpectrumLibrary(SpectraLibrary):
    """
    A library to manage and store scattering spectra data.
    """

    def __init__(self, additional_folder_path: str = None):
        """
        Initializes the ScatteringSpectrumLibrary with scattering spectra data.

        :param additional_folder_path: An additional folder path for more scattering spectra data.
        """
        super(ScatteringSpectrumLibrary, self).__init__("scattering_spectra_data", additional_folder_path)

    @staticmethod
    def CONSTANT_SCATTERING_ARBITRARY(scattering: float = 1) -> Spectrum:
        """
        Creates a constant scattering spectrum with an arbitrary value.

        :param scattering: The scattering value to use.
        :return: A Spectrum instance with constant scattering.
        """
        return Spectrum("Constant Scattering (arb)", np.asarray([450, 1000]),
                        np.asarray([scattering, scattering]))

    @staticmethod
    def scattering_from_rayleigh_and_mie_theory(name: str, mus_at_500_nm: float = 1.0,
                                                fraction_rayleigh_scattering: float = 0.0,
                                                mie_power_law_coefficient: float = 0.0) -> Spectrum:
        """
        Creates a scattering spectrum based on Rayleigh and Mie scattering theory.

        :param name: The name of the spectrum.
        :param mus_at_500_nm: Scattering coefficient at 500 nm.
        :param fraction_rayleigh_scattering: Fraction of Rayleigh scattering.
        :param mie_power_law_coefficient: Power law coefficient for Mie scattering.
        :return: A Spectrum instance based on Rayleigh and Mie scattering theory.
        """
        wavelengths = np.arange(450, 1001, 1)
        scattering = (mus_at_500_nm * (fraction_rayleigh_scattering * (wavelengths / 500) ** 1e-4 +
                      (1 - fraction_rayleigh_scattering) * (wavelengths / 500) ** -mie_power_law_coefficient))
        return Spectrum(name, wavelengths, scattering)


class AbsorptionSpectrumLibrary(SpectraLibrary):
    """
    A library to manage and store absorption spectra data.
    """

    def __init__(self, additional_folder_path: str = None):
        """
        Initializes the AbsorptionSpectrumLibrary with absorption spectra data.

        :param additional_folder_path: An additional folder path for more absorption spectra data.
        """
        super(AbsorptionSpectrumLibrary, self).__init__("absorption_spectra_data", additional_folder_path)

    @staticmethod
    def CONSTANT_ABSORBER_ARBITRARY(absorption_coefficient: float = 1) -> Spectrum:
        """
        Creates a constant absorption spectrum with an arbitrary value.

        :param absorption_coefficient: The absorption coefficient to use.
        :return: A Spectrum instance with constant absorption.
        """
        return Spectrum("Constant Absorber (arb)", np.asarray([450, 1000]),
                        np.asarray([absorption_coefficient, absorption_coefficient]))


class RefractiveIndexSpectrumLibrary(SpectraLibrary):

    def __init__(self, additional_folder_path: str = None):
        super(RefractiveIndexSpectrumLibrary, self).__init__("refractive_index_spectra_data", additional_folder_path)

    @staticmethod
    def CONSTANT_REFRACTOR_ARBITRARY(refractive_index: float = 1):
        return Spectrum("Constant Refractor (arb)", np.asarray([450, 1000]),
                        np.asarray([refractive_index, refractive_index]))


def get_simpa_internal_absorption_spectra_by_names(absorption_spectrum_names: list):
    """
    Retrieves SIMPA internal absorption spectra by their names.

    :param absorption_spectrum_names: List of absorption spectrum names to retrieve.
    :return: List of Spectrum instances corresponding to the given names.
    """
    lib = AbsorptionSpectrumLibrary()
    spectra = []
    for spectrum_name in absorption_spectrum_names:
        spectra.append(lib.get_spectrum_by_name(spectrum_name))
    return spectra


def view_saved_spectra(save_path=None, mode="absorption"):
    """
    Opens a matplotlib plot and visualizes the available spectra.

    :param save_path: If not None, then the figure will be saved as a PNG file to the destination.
    :param mode: Specifies the type of spectra to visualize ("absorption", "scattering", "anisotropy" or "refractive_index).
    """
    plt.figure(figsize=(11, 8))
    if mode == "absorption":
        for spectrum in AbsorptionSpectrumLibrary():
            plt.semilogy(spectrum.wavelengths,
                         spectrum.values,
                         label=spectrum.spectrum_name)
    elif mode == "scattering":
        for spectrum in ScatteringSpectrumLibrary():
            plt.semilogy(spectrum.wavelengths,
                         spectrum.values,
                         label=spectrum.spectrum_name)
    elif mode == "anisotropy":
        for spectrum in AnisotropySpectrumLibrary():
            plt.semilogy(spectrum.wavelengths,
                         spectrum.values,
                         label=spectrum.spectrum_name)
    elif mode == "refractive_index":
        for spectrum in RefractiveIndexSpectrumLibrary():
            plt.semilogy(spectrum.wavelengths,
                         spectrum.values,
                         label=spectrum.spectrum_name)
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Choose from 'absorption', 'scattering', 'anisotropy' or 'refractive_index'.")

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
