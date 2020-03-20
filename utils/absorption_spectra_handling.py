import numpy as np
import glob

ABSORPTION_SPECTRA_FILE_NAME = "ippai_absorption_spectra.npz"


class AbsorptionSpectrum(object):
    """
    An instance of this class represents the absorption spectrum over wavelength for a particular
    """

    def __init__(self, spectrum_name: str, wavelengths: np.ndarray, absorption_per_centimeter: np.ndarray):
        """
        @param spectrum_name:
        @param wavelengths:
        @param absorption_per_centimeter:
        """
        self.spectrum_name = spectrum_name
        self.wavelengths = wavelengths
        self.absorption_per_centimeter = absorption_per_centimeter

        if np.shape(wavelengths) != np.shape(absorption_per_centimeter):
            raise ValueError("The shape of the wavelengths and the absorption coefficients did not match!")

    def get_absorption_over_wavelength(self):
        return np.asarray([self.wavelengths, self.absorption_per_centimeter])

    def get_absorption_for_wavelength(self, wavelength: float) -> float:
        """
        @param wavelength: the wavelength to retrieve a optical absorption value for [cm^{-1}].
        @return: the best matching linearly interpolated absorption value for the given wavelength.
        """
        # TODO: behavior outside of the available wavelength range?
        if wavelength > max(self.wavelengths):
            raise ValueError("Given wavelength is larger than the maximum available wavelength")
        if wavelength < min(self.wavelengths):
            raise ValueError("Given wavelength is smaller than the minimum available wavelength")

        return np.interp(wavelength, self.wavelengths, self.absorption_per_centimeter)


def load_absorption_spectra_numpy_array(path: str = None) -> [np.lib.npyio.NpzFile, str]:
    """
    @param path: A search path to look for the spectra file.
    """

    if path is None:
        path = ""

    # A list of plausible search paths where the npz file could be
    search_paths = [
        path + "**/" + ABSORPTION_SPECTRA_FILE_NAME,
        "**/" + ABSORPTION_SPECTRA_FILE_NAME,
        "/**/" + ABSORPTION_SPECTRA_FILE_NAME,
        path + "**/data/" + ABSORPTION_SPECTRA_FILE_NAME,
        "**/data/" + ABSORPTION_SPECTRA_FILE_NAME,
        "/**/data/" + ABSORPTION_SPECTRA_FILE_NAME
    ]

    # If the path is already the full path to the file, include it into the search list at first position.
    # In case it exists this will drastically shorten the search time.
    if ABSORPTION_SPECTRA_FILE_NAME in path:
        search_paths.insert(0, path)

    file = []
    for search_path in search_paths:
        file = glob.glob(search_path)
        if (file is not None) and (len(file) > 0):
            break

    if (file is None) or (len(file) == 0):
        raise FileNotFoundError("Did not find '" + ABSORPTION_SPECTRA_FILE_NAME + "'.")

    return [np.load(file[0], allow_pickle=True), file]

