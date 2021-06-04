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
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.simulation_components import ProcessingComponent
from simpa.utils.libraries.spectra_library import SPECTRAL_LIBRARY
import numpy as np
import scipy.linalg as linalg
from simpa.utils.settings import Settings


class LinearUnmixingProcessingComponent(ProcessingComponent):
    """
        Performs linear spectral unmixing using FLUPAI on the defined data field and
        saves a dictionary containing the chromophore concentration for each chromophore
        specified in the settings and (if selected) the blood oxygen saturation.

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
        super(ProcessingComponent, self).__init__(global_settings=global_settings)

        self.global_settings = global_settings
        self.component_settings = Settings(global_settings[component_settings_key])

        self.data_array = []
        self.global_wavelengths = []
        self.chromophore_wavelengths_dict = {}
        self.chromophore_spectra_dict = {}
        self.pseudo_inverse_absorption_matrix = []
        self.chromophore_concentrations = []
        self.chromophore_concentrations_dict = {}

    def run(self):

        self.logger.info("Performing linear spectral unmixing...")

        # check component settings to confirm completeness and load tags
        if Tags.DATA_FIELD not in self.component_settings.keys():
            self.logger.critical()
            raise KeyError(f"The tag Tags.DATA_FIELD must be set in order to perform linear unmixing!")
        else:
            self.logger.info(f"Linear unmixing will be performed on data field: {self.component_settings[Tags.DATA_FIELD]}.")
            data_field = self.component_settings[Tags.DATA_FIELD]

        if Tags.WAVELENGTHS in self.component_settings.keys():
            self.global_wavelengths = self.component_settings[Tags.WAVELENGTHS]
        else:
            self.global_wavelengths = self.global_settings[Tags.WAVELENGTHS]

        # compute volume dimensions to create data array
        spacing = self.global_settings[Tags.SPACING_MM]
        x_dim = int(self.global_settings[Tags.DIM_VOLUME_X_MM] / spacing)
        y_dim = int(self.global_settings[Tags.DIM_VOLUME_Y_MM] / spacing)
        z_dim = int(self.global_settings[Tags.DIM_VOLUME_Z_MM] / spacing)

        # create array which contains the simulated data fields of all specified wavelengths
        # the first dimension of the created data array encodes wavelength
        self.data_array = np.empty((len(self.global_wavelengths), x_dim, y_dim, z_dim))
        for i in range(len(self.global_wavelengths)):
            self.data_array[i, :, :, :] = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                                                          data_field,
                                                          self.global_wavelengths[i])

        # get absorption values for all chromophores and wavelengths using SIMPAs spectral library
        self.build_chromophore_spectra_dict()

        if self.chromophore_spectra_dict == {}:
            raise KeyError("Linear unmixing must be performed for at least one chromophore. "
                           "Please specify at least one chromophore in the component settings by setting "
                           "the corresponding tag.")

        # create the pseudo inverse absorption matrix
        self.pseudo_inverse_absorption_matrix = self.create_piv_absorption_matrix()
        self.logger.debug(f"The pseudo inverse absorption matrix has shape {np.shape(self.pseudo_inverse_absorption_matrix)}.")

        # perform fast linear unmixing
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
        Function build the chromophore spectra dictionary for each chromophore using SIMPAs spectral library.
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

    def create_chromophore_spectra_entry(self, chromophore_tag, chromophore_name):
        """
        Builds entry containing spectra for a chromophore specified by tag and name and saves it in
        chromophore_spectra_dict. The name must match the ones used in the spectral library of SIMPA.
        """
        if len(self.component_settings[chromophore_tag]) < 2:
            self.logger.critical(f"Linear unmixing should be performed with at least two wavelengths! "
                                 f"Unmixing is approached with just {len(self.component_settings[chromophore_tag])} "
                                 f"wavelength for {chromophore_name}.")

        # TODO: change error handling e.g. (try, except)
        for wavelength in self.component_settings[chromophore_tag]:
            if wavelength not in self.global_wavelengths:
                self.logger.critical(f"{chromophore_name}: wavelength {wavelength}nm was not simulated and will not be used.")
            if chromophore_name != "Nickel Sulphide" and chromophore_name != "Copper Sulphide":
                if wavelength > 1000 or wavelength < 450:
                    raise ValueError(f"{chromophore_name}: wavelength {wavelength}nm is larger/smaller than allowed.")
            else:
                if wavelength > 980 or wavelength < 500:
                    raise ValueError(f"{chromophore_name}: wavelength {wavelength}nm is larger/smaller than allowed.")

        self.chromophore_wavelengths_dict[chromophore_name] = self.component_settings[chromophore_tag]
        spectra = SPECTRAL_LIBRARY.get_spectrum_by_name(chromophore_name)
        self.chromophore_spectra_dict[chromophore_name] = [spectra.get_absorption_for_wavelength(wavelength)
                                                           for wavelength in self.component_settings[chromophore_tag]]
        return None

    def create_piv_absorption_matrix(self):
        """
        Method that returns the pseudo inverse of the  absorption (endmember) matrix
        needed for linear unmixing.

        :return: pseudo inverse absorption matrix
        """

        numberWavelengths = len(self.global_wavelengths)
        numberChromophores = len(self.chromophore_spectra_dict.keys())

        # prepare matrix
        endmemberMatrix = np.zeros((numberWavelengths, numberChromophores))

        # write absorption data for each chromophore and the corresponding wavelength into an array (matrix)
        for index, key in enumerate(self.chromophore_spectra_dict.keys()):
            for wave in range(numberWavelengths):
                if self.global_wavelengths[wave] in self.chromophore_wavelengths_dict[key]:
                    endmemberMatrix[wave][index] = self.chromophore_spectra_dict[key][self.chromophore_wavelengths_dict[key].index(self.global_wavelengths[wave])]

        return linalg.pinv(endmemberMatrix)

    def flupai(self):
        """
        Fast Linear Unmixing for PhotoAcoustic Imaging (FLUPAI) is based on
        SVD decomposition with a pseudo inverse, which is equivalent to a least squares
        ansatz for linear spectral unmixing of multi-spectral photoacoustic images.

        :return: list with unmixed images containing the chromophore concentration.
        :raise: SystemExit.
        """

        # reshape image data to [number of wavelength, number of pixel]
        dims_raw = np.shape(self.data_array)
        try:
            reshapedData = np.reshape(self.data_array, (dims_raw[0], -1))
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
        chromophores = []
        for chromophore in range(numberChromophores):
            chromophores.append(np.reshape(output[chromophore, :], (dims_raw[1], dims_raw[2], dims_raw[3])))
        return chromophores

    def calculate_sO2(self):
        """
        Function calculates sO2 (blood oxygen saturation) values for given concentrations of oxyhemoglobin and deoxyhemoglobin.
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

