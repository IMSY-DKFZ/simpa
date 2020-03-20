from utils import AbsorptionSpectrum
import numpy as np
import matplotlib.pylab as plt


class AbsorptionSpectrumLibrary(object):

    def __init__(self):
        self.spectra = [
            AbsorptionSpectrum("Deoxyhemoglobin", np.arange(400, 1000, 1), np.arange(400, 1000, 1)),
            AbsorptionSpectrum("Oxyhemoglobin", np.arange(400, 1000, 1), np.arange(400, 1000, 1)),
            AbsorptionSpectrum("Water", np.arange(400, 1000, 1), np.arange(400, 1000, 1)),
            AbsorptionSpectrum("Fat", np.arange(400, 1000, 1), np.arange(400, 1000, 1)),
            AbsorptionSpectrum("Melanin", np.arange(400, 1000, 1), np.arange(400, 1000, 1))
        ]
        self.i = len(self.spectra)

    def __next__(self):
        if self.i > 0:
            self.i -= 1
            return self.spectra[self.i]
        raise StopIteration()

    def __iter__(self):
        return self

    def get_spectra_names(self):
        return [spectrum.spectrum_name for spectrum in self]

    def get_spectrum_by_name(self, spectrum_name: str) -> AbsorptionSpectrum:
        for spectrum in self:
            if spectrum.spectrum_name == spectrum_name:
                return spectrum

        raise LookupError("No spectrum for the given name exists")


SPECTRAL_LIBRARY = AbsorptionSpectrumLibrary()


def view_absorption_spectra():
    plt.figure()
    for spectrum in SPECTRAL_LIBRARY:
        plt.plot(spectrum.get_absorption_over_wavelength().T)
    plt.show()
    plt.close()