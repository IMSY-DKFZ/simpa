"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.io_handling import load_data_field, save_data_field
from simpa.algorithms.multispectral import MultispectralProcessingAlgorithm
from simpa.utils.libraries.spectra_library import SPECTRAL_LIBRARY
import numpy as np
import scipy.linalg as linalg
from simpa.utils.settings import Settings


class LinearUnmixingProcessingComponent(MultispectralProcessingAlgorithm):
    """
        Performs linear spectral unmixing (LU) using Fast Linear Unmixing for PhotoAcoustic Imaging (FLUPAI)
        on the defined data field for each chromophore specified in the component settings.

        This component saves a dictionary containing the chromophore concentrations and corresponding wavelengths for
        each chromophore. If the tag LINEAR_UNMIXING_COMPUTE_SO2 is set True the blood oxygen saturation
        is saved as well, however, this is only possible if the chromophores oxy- and deoxyhemoglobin are specified.
        IMPORTANT:
        Linear unmixing should only be performed with at least two wavelengths:
        e.g. Tags.LINEAR_UNMIXING_OXYHEMOGLOBIN: [750, 800]

        :param kwargs:
           **Tags.DATA_FIELD (required)
           **Tags.LINEAR_UNMIXING_OXYHEMOGLOBIN
           **Tags.LINEAR_UNMIXING_DEOXYHEMOGLOBIN
           **Tags.LINEAR_UNMIXING_WATER
           **Tags.LINEAR_UNMIXING_FAT
           **Tags.LINEAR_UNMIXING_MELANIN
           **Tags.LINEAR_UNMIXING_NICKEL_SULPHITE
           **Tags.LINEAR_UNMIXING_COPPER_SULPHITE
           **Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_ZERO
           **Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_ONE
           **Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_TEN
           **Tags.LINEAR_UNMIXING_COMPUTE_SO2 (default: False)
           **settings (required)
           **component_settings_key (required)
        """

    def __init__(self, global_settings, component_settings_key: str):
        super(LinearUnmixingProcessingComponent, self).__init__(global_settings=global_settings,
                                                                component_settings_key=component_settings_key)

        self.chromophore_spectra_dict = {}  # dictionary containing the spectrum for each chromophore and wavelength
        self.pseudo_inverse_absorption_matrix = []  # endmember matrix needed in LU

        self.chromophore_concentrations = []  # list of LU results
        self.chromophore_concentrations_dict = {}  # dictionary of LU results
        self.chromophore_wavelengths_dict = {}  # dictionary of corresponding wavelengths

    def run(self):

        self.logger.info("Performing linear spectral unmixing...")

        # get absorption values for all chromophores and wavelengths using SIMPAs spectral library
        # the absorption values are saved in self.chromophore_spectra_dict
        # e.g. {'Oxyhemoglobin': [2.77, 5.67], ...}
        # the corresponding wavelengths for each chromophore are saved in self.chromophore_wavelengths_dict
        # e.g. {'Oxyhemoglobin': [750, 850], ...}
        self.build_chromophore_spectra_dict()

        # check if absorption dictionary contains any spectra
        if self.chromophore_spectra_dict == {}:
            raise KeyError("Linear unmixing must be performed for at least one chromophore. "
                           "Please specify at least one chromophore in the component settings by setting "
                           "the corresponding tag.")

        # create the pseudo inverse absorption matrix needed by FLUPAI
        # the matrix should have the shape [#chromophores, #global wavelengths]
        self.pseudo_inverse_absorption_matrix = self.create_piv_absorption_matrix()
        self.logger.debug(f"The pseudo inverse absorption matrix has shape {np.shape(self.pseudo_inverse_absorption_matrix)}.")

        # perform fast linear unmixing FLUPAI
        # the result saved in self.chromophore_concentrations is a list with the unmixed images
        # containing the chromophore concentration
        self.chromophore_concentrations = self.flupai()
        self.logger.debug(f"The unmixing result has shape {np.shape(self.chromophore_concentrations)}.")

        # split results to create dictionary which contains linear unmixing result for each chromophore
        for index, chromophore in enumerate(self.chromophore_spectra_dict.keys()):
            self.chromophore_concentrations_dict[chromophore] = self.chromophore_concentrations[index]
        self.logger.info(f"The chromophore concentration was computed for chromophores: "
                          f"{self.chromophore_concentrations_dict.keys()}")

        # compute blood oxygen saturation if selected
        if Tags.LINEAR_UNMIXING_COMPUTE_SO2 in self.component_settings:
            if self.component_settings[Tags.LINEAR_UNMIXING_COMPUTE_SO2]:
                self.logger.info("Blood oxygen saturation is calculated and saved.")
                sO2 = self.calculate_sO2()
                save_dict = {
                    "sO2": sO2,
                    "chromophore_concentrations": self.chromophore_concentrations_dict,
                    "chromophore_wavelengths": self.chromophore_wavelengths_dict
                }
            else:
                self.logger.info("Blood oxygen saturation is not calculated.")
                save_dict = {
                    "chromophore_concentrations": self.chromophore_concentrations_dict,
                    "chromophore_wavelengths": self.chromophore_wavelengths_dict
                }
        else:
            self.logger.info("Blood oxygen saturation is not calculated.")
            save_dict = {
                "chromophore_concentrations": self.chromophore_concentrations_dict,
                "chromophore_wavelengths": self.chromophore_wavelengths_dict
            }

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

        if Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_TEN in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_TEN, "Constant Absorber (10)")
        if Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_ONE in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_ONE, "Constant Absorber (1)")
        if Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_ZERO in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_CONSTANT_ABSORBER_ZERO, "Constant Absorber (0)")
        if Tags.LINEAR_UNMIXING_COPPER_SULPHIDE in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_COPPER_SULPHIDE, "Copper Sulphide")
        if Tags.LINEAR_UNMIXING_NICKEL_SULPHIDE in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_NICKEL_SULPHIDE, "Nickel Sulphide")
        if Tags.LINEAR_UNMIXING_MELANIN in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_MELANIN, "Melanin")
        if Tags.LINEAR_UNMIXING_FAT in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_FAT, "Fat")
        if Tags.LINEAR_UNMIXING_WATER in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_WATER, "Water")
        if Tags.LINEAR_UNMIXING_OXYHEMOGLOBIN in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_OXYHEMOGLOBIN, "Oxyhemoglobin")
        if Tags.LINEAR_UNMIXING_DEOXYHEMOGLOBIN in self.component_settings:
            self.create_chromophore_spectra_entry(Tags.LINEAR_UNMIXING_DEOXYHEMOGLOBIN, "Deoxyhemoglobin")

        return None

    def create_chromophore_spectra_entry(self, chromophore_tag: tuple, chromophore_name: str):
        """
        This function builds the spectra for a chromophore specified by tag and name and saves it in
        self.chromophore_spectra_dict and creates a dictionary containing the corresponding wavelengths.
        The name must match the ones used in the spectral library of SIMPA.
        """
        if len(self.component_settings[chromophore_tag]) < 2:
            self.logger.critical(f"Linear unmixing should be performed with at least two wavelengths! "
                                 f"Unmixing is approached with just {len(self.component_settings[chromophore_tag])} "
                                 f"wavelength for {chromophore_name}.")
        # TODO: refactor of Spectra Library and error handling
        try:
            self.chromophore_wavelengths_dict[chromophore_name] = self.component_settings[chromophore_tag]
            spectra = SPECTRAL_LIBRARY.get_spectrum_by_name(chromophore_name)
            self.chromophore_spectra_dict[chromophore_name] = [spectra.get_value_for_wavelength(wavelength)
                                                                for wavelength in self.component_settings[chromophore_tag]]
        except Exception as e:
            self.logger.warning("Loading of spectrum not successful.")
            self.logger.debug(e)
            raise ValueError("For details see above.")

    def create_piv_absorption_matrix(self) -> np.ndarray:
        """
        Method that returns the pseudo inverse of the absorption (endmember) matrix
        needed for linear unmixing.

        :return: pseudo inverse absorption matrix
        """

        numberWavelengths = len(self.wavelengths)
        numberChromophores = len(self.chromophore_spectra_dict.keys())

        # prepare matrix
        endmemberMatrix = np.zeros((numberWavelengths, numberChromophores))

        # write absorption data for each chromophore and the corresponding wavelength into an array (matrix)
        for index, key in enumerate(self.chromophore_spectra_dict.keys()):
            for wave in range(numberWavelengths):
                if self.wavelengths[wave] in self.chromophore_wavelengths_dict[key]:
                    endmemberMatrix[wave][index] = self.chromophore_spectra_dict[key][
                        self.chromophore_wavelengths_dict[key].index(self.wavelengths[wave])]

        return linalg.pinv(endmemberMatrix)

    def flupai(self) -> list:
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

        # matmul of x = PI * b with x chromophore information, PI pseudo inverse with absorber information and b
        # containing the measured pixel
        try:
            output = np.matmul(self.pseudo_inverse_absorption_matrix, reshapedData)
        except Exception:
            self.logger.critical(f"Matrix multiplication failed probably caused by mismatching dimensions of pseudo"
                                 f"inverse ({len(self.pseudo_inverse_absorption_matrix[0])}) and "
                                 f"input data ({dims_raw[0]})!")
            raise ValueError("Pseudo inverse (PI) and input data (b) must have matching sizes...")

        # write output into list of images containing the chromophore information
        numberChromophores = len(self.pseudo_inverse_absorption_matrix)
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
