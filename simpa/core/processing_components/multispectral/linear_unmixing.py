# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.io_handling import save_data_field
from simpa.core.processing_components.multispectral import MultispectralProcessingAlgorithm
from simpa.utils.libraries.spectrum_library import Spectrum
import numpy as np
import scipy.linalg as linalg
from scipy.optimize import nnls


class LinearUnmixing(MultispectralProcessingAlgorithm):
    """
        Performs linear spectral unmixing (LU) using Fast Linear Unmixing for PhotoAcoustic Imaging (FLUPAI)
        on the defined data field for each chromophore specified in the component settings.

        If tag LINEAR_UNMIXING_NON_NEGATIVE is set to True non-negative linear unmixing is performed, which solves the
        KKT (Karush-Kuhn-Tucker) conditions for the non-negative least squares problem.

        This component saves a dictionary containing the chromophore concentrations and corresponding wavelengths for
        each chromophore. If the tag LINEAR_UNMIXING_COMPUTE_SO2 is set True the blood oxygen saturation
        is saved as well, however, this is only possible if the chromophores oxy- and deoxyhemoglobin are specified.
        IMPORTANT:
        Linear unmixing should only be performed with at least two wavelengths:
        e.g. Tags.WAVELENGTHS: [750, 800]

        :param kwargs:
           **Tags.DATA_FIELD (required)
           **Tags.LINEAR_UNMIXING_SPECTRA (required)
           **Tags.WAVELENGTHS (default: None, if None, then settings[Tags.WAVELENGTHS] will be used.)
           **Tags.LINEAR_UNMIXING_COMPUTE_SO2 (default: False)
           **Tags.LINEAR_UNMIXING_NON_NEGATIVE (default: False)
           **settings (required)
           **component_settings_key (required)
        """

    def __init__(self, global_settings, component_settings_key: str):
        super(LinearUnmixing, self).__init__(global_settings=global_settings,
                                             component_settings_key=component_settings_key)

        self.chromophore_spectra_dict = {}  # dictionary containing the spectrum for each chromophore and wavelength
        self.absorption_matrix = []  # endmember matrix needed in LU
        self.pseudo_inverse_absorption_matrix = []

        self.chromophore_concentrations = []  # list of LU results
        self.chromophore_concentrations_dict = {}  # dictionary of LU results
        self.wavelengths = []  # list of wavelengths

    def run(self):

        self.logger.info("Performing linear spectral unmixing...")

        if Tags.WAVELENGTHS in self.component_settings:
            self.wavelengths = self.component_settings[Tags.WAVELENGTHS]
        else:
            if Tags.WAVELENGTHS in self.global_settings:
                self.wavelengths = self.global_settings[Tags.WAVELENGTHS]
            else:
                msg = "Was not able to get wavelengths from component_settings or global_settings."
                self.logger.critical(msg)
                raise AssertionError(msg)

        if len(self.wavelengths) < 2:
            msg = "Linear unmixing should be performed with at least two wavelengths!"
            self.logger.critical(msg)
            raise AssertionError(msg)

        # Build internal list of spectra based on Tags.LINEAR_UNMIXING_SPECTRA
        self.build_chromophore_spectra_dict()

        # check if absorption dictionary contains any spectra
        if self.chromophore_spectra_dict == {}:
            raise KeyError("Linear unmixing must be performed for at least one chromophore. "
                           "Please specify at least one chromophore in the component settings by setting "
                           "the corresponding tag.")

        # check if non-negative contraint should be used for linear unmixing
        non_negative = False
        if Tags.LINEAR_UNMIXING_NON_NEGATIVE in self.component_settings:
            non_negative = Tags.LINEAR_UNMIXING_NON_NEGATIVE

        # create the absorption matrix needed by FLUPAI
        # the matrix should have the shape [#global wavelengths, #chromophores]
        self.absorption_matrix = self.create_absorption_matrix()
        self.logger.debug(f"The absorption matrix has shape {np.shape(self.absorption_matrix)}.")

        # perform fast linear unmixing FLUPAI
        # the result saved in self.chromophore_concentrations is a list with the unmixed images
        # containing the chromophore concentration
        self.chromophore_concentrations = self.flupai(non_negative=non_negative)
        self.logger.debug(f"The unmixing result has shape {np.shape(self.chromophore_concentrations)}.")

        # split results to create dictionary which contains linear unmixing result for each chromophore
        for index, chromophore in enumerate(self.chromophore_spectra_dict.keys()):
            self.chromophore_concentrations_dict[chromophore] = self.chromophore_concentrations[index]
        self.logger.info(f"The chromophore concentration was computed for chromophores: "
                          f"{self.chromophore_concentrations_dict.keys()}")

        # compute blood oxygen saturation if selected
        save_dict = {
                    "chromophore_concentrations": self.chromophore_concentrations_dict,
                    "wavelengths": self.wavelengths
                    }
        if Tags.LINEAR_UNMIXING_COMPUTE_SO2 in self.component_settings:
            if self.component_settings[Tags.LINEAR_UNMIXING_COMPUTE_SO2]:
                self.logger.info("Blood oxygen saturation is calculated and saved.")
                save_dict["sO2"] = self.calculate_sO2()

        # save linear unmixing result in hdf5
        save_data_field(save_dict, self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                        Tags.LINEAR_UNMIXING_RESULT, wavelength=None)

        self.logger.info("Performing linear spectral unmixing......[Done]")

    def build_chromophore_spectra_dict(self):
        """
        This function builds the absorption spectra dictionary for each chromophore using SIMPAs spectral library
        and saves the result in self.chromophore_spectra_dict.
        This function might have to change drastically if the design of the spectral library changes in the future!
        """

        if Tags.LINEAR_UNMIXING_SPECTRA in self.component_settings:
            spectra = self.component_settings[Tags.LINEAR_UNMIXING_SPECTRA]
            if len(spectra) < 2:
                raise AssertionError(f"Need at least two endmembers for unmixing! You provided {len(spectra)}.")
            for spectrum in spectra:
                self.create_chromophore_spectra_entry(spectrum)
        else:
            raise AssertionError("Tried to unmix without spectra definitions. Make sure that the"
                                 " Tags.LINEAR_UNMIXING_SPECTRA tag is set in the linear unmixing settings.")

    def create_chromophore_spectra_entry(self, spectrum: Spectrum):
        """
        This function builds the spectra for a chromophore specified by tag and name and saves it in
        self.chromophore_spectra_dict and creates a dictionary containing the corresponding wavelengths.
        The name must match the ones used in the spectral library of SIMPA.
        """

        self.chromophore_spectra_dict[spectrum.spectrum_name] = [spectrum.get_value_for_wavelength(wavelength)
                                                                 for wavelength in self.wavelengths]

    def create_absorption_matrix(self) -> np.ndarray:
        """
        Method that returns the absorption (endmember) matrix needed for linear unmixing.

        :return: absorption matrix
        """

        numberWavelengths = len(self.wavelengths)
        numberChromophores = len(self.chromophore_spectra_dict.keys())

        # prepare matrix
        endmemberMatrix = np.zeros((numberWavelengths, numberChromophores))

        # write absorption data for each chromophore and the corresponding wavelength into an array (matrix)
        for index, key in enumerate(self.chromophore_spectra_dict.keys()):
            for wave in range(numberWavelengths):
                    endmemberMatrix[wave][index] = self.chromophore_spectra_dict[key][wave]

        return endmemberMatrix

    def flupai(self, non_negative=False) -> list:
        """
        Fast Linear Unmixing for PhotoAcoustic Imaging (FLUPAI) is based on
        SVD decomposition with a pseudo inverse, which is equivalent to a least squares
        ansatz for linear spectral unmixing of multi-spectral photoacoustic images.

        :return: list with unmixed images containing the chromophore concentration.
        :raise: SystemExit.
        """

        # reshape image data to [number of wavelength, number of pixel]
        dims_raw = np.shape(self.data)
        try:
            reshapedData = np.reshape(self.data, (dims_raw[0], -1))
        except Exception:
            self.logger.critical(f"FLUPAI failed probably caused by wrong input dimensions of {dims_raw}!")
            raise ValueError("Reshaping of input data failed. FLUPAI expects a 4 dimensional numpy array, "
                             "where the first dimension represents the wavelengths and the second, third and fourth "
                             "dimension are representing a single wavelength PA image.")

        # if non_negative is False, matmul of x = PI * b with x chromophore information,
        # PI pseudo inverse with absorber information and b containing the measured pixel,
        # else non-negative least squares is performed.
        try:
            if non_negative:
                output = []
                for i in range(np.shape(reshapedData)[1]):
                    foo, ris = nnls(np.array(self.absorption_matrix), reshapedData[:, i])
                    output.append(foo)

                output = np.swapaxes(output, axis1=0, axis2=1)
            else:
                self.pseudo_inverse_absorption_matrix = linalg.pinv(self.absorption_matrix)
                output = np.matmul(self.pseudo_inverse_absorption_matrix, reshapedData)

        except Exception as e:
            self.logger.critical(f"Matrix multiplication failed probably caused by mismatching dimensions of absorption"
                                 f"matrix ({len(self.absorption_matrix[1])}) and "
                                 f"input data ({dims_raw[0]})!")
            print(e)
            raise ValueError("Absorption matrix and input data must have matching sizes...")

    # write output into list of images containing the chromophore information
        numberChromophores = np.shape(output)[0]
        chromophores_concentrations = []
        for chromophore in range(numberChromophores):
            chromophores_concentrations.append(np.reshape(output[chromophore, :], (dims_raw[1:])))
        return chromophores_concentrations

    def calculate_sO2(self) -> np.ndarray:
        """
        Function calculates sO2 (blood oxygen saturation) values for given concentrations
        of oxyhemoglobin and deoxyhemoglobin. Of course this is only possible if the concentrations of both
        chromophores were calculated by this component/were specified in settings.
        """

        try:
            concentration_oxy = self.chromophore_concentrations_dict["Oxyhemoglobin"]
            concentration_deoxy = self.chromophore_concentrations_dict["Deoxyhemoglobin"]

            sO2 = concentration_oxy / (concentration_oxy + concentration_deoxy)
            # if total hemoglobin is zero handle NaN by setting sO2 to zero
            where_are_NaNs = np.isnan(sO2)
            sO2[where_are_NaNs] = 0
            return sO2

        except Exception:
            raise KeyError("Chromophores oxy- and/or deoxyhemoglobin were not specified in component settings, "
                           "so so2 cannot be calculated!")
