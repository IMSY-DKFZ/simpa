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
from simpa.utils import EPS
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.simulation_components import ProcessingComponent
from simpa.utils.libraries.spectra_library import SPECTRAL_LIBRARY

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.linalg as linalg
import pandas as pd


class LinearUnmixingProcessingComponent(ProcessingComponent):
    """
        Performs linear spectral unmixing on the defined data field.
        :param kwargs:
        :: TODO :: Which settings are required
               # chromophores and wavelength per chromophore
           **data_field (required)
        """

    def run(self):

        self.logger.info("Linear Unmixing...")

        if Tags.DATA_FIELD not in self.component_settings.keys():
            self.logger.critical()
            raise KeyError(f"The field {Tags.DATA_FIELD} must be set in order to use the gaussian_noise field.")

        data_field = self.component_settings[Tags.DATA_FIELD]
        wavelengths = self.global_settings[Tags.WAVELENGTHS]
        chromo_dict = self.component_settings[Tags.LINEAR_UNMIXING_CHROMOPHORE_DICT]

        spacing = self.global_settings[Tags.SPACING_MM]
        x_dim = int(self.global_settings[Tags.DIM_VOLUME_X_MM] / spacing)
        y_dim = int(self.global_settings[Tags.DIM_VOLUME_Y_MM] / spacing)
        z_dim = int(self.global_settings[Tags.DIM_VOLUME_Z_MM] / spacing)

        data_array = np.empty((len(wavelengths), x_dim, y_dim, z_dim))
        for i in range(len(wavelengths)):
            data_array[i, :, :, :] = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelengths[i])

        print("Wavelengths:", wavelengths)
        print("Chromo_dict:", chromo_dict)

        ### TODO: except when wavelength is larger/smaller than allowed
        chromo_absorption_dict = {}
        for chromophore in chromo_dict.keys():
            if chromophore == "oxy":
                spectra = SPECTRAL_LIBRARY.get_spectrum_by_name("Oxyhemoglobin")
                chromo_absorption_dict["oxy"] = [spectra.get_absorption_for_wavelength(wavelength) for wavelength in chromo_dict[chromophore]]
            elif chromophore == "deoxy":
                spectra = SPECTRAL_LIBRARY.get_spectrum_by_name("Deoxyhemoglobin")
                chromo_absorption_dict["deoxy"] = [spectra.get_absorption_for_wavelength(wavelength) for wavelength in chromo_dict[chromophore]]

        print("Abs_dict:", chromo_absorption_dict)

        pinv = create_piv_absorption_matrix(chromo_dict, wavelengths, chromo_absorption_dict)
        print("shape pinv:", np.shape(pinv))
        unmixing_result = flupai(pinv, data_array)
        print("shape un result:", np.shape(unmixing_result))

        result = unmixing_result[0][:, :, 4] / (unmixing_result[0][:, :, 4] + unmixing_result[1][:, :, 4])
        print("shape result:", np.shape(result))
        plt.imshow(result)
        plt.colorbar()
        plt.show()
g
        # testing
        #print('Shape data_array: ', np.shape(data_array))

        sys.exit('Copy of noise processing. +++ WIP +++ ')
        meaningfull_name = 'test'
        #save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], meaningfull_name,)

        self.logger.info("Applying Gaussian Noise Model...[Done]")


# +++++ COPIED FROM CMD TOOLS


def create_piv_absorption_matrix(chromo_dict, wavelengths, chromo_absorption_dict):
    """ Method that returns a str list of choosen chromophores, which are read out of a xml file as well as the
    pseudo inverse of the  absorption (endmember) matrix.

    :param xmlFile: name.options.xml containing the spectral unmxiing settings
    :type xmlFile: str
    :param spec: name of spectrum file as pandas data frame pickle
    :type spec: str
    :return: pseudo inverse, str. list of chromophores in 'correct' order
    """
    # read spectrum and xml file
    numberWavelengths = len(wavelengths)

    numberChromophores = len(chromo_dict.keys())
    endmemberMatrix = np.zeros((numberWavelengths, numberChromophores))

    # write absorption data fpr each chromophore and the corresponding wavelength into an array (matrix)
    for index, key in enumerate(chromo_dict.keys()):
        for wave in range(numberWavelengths):
            if wavelengths[wave] in chromo_dict[key]:
                endmemberMatrix[wave][index] = chromo_absorption_dict[key][chromo_dict[key].index(wavelengths[wave])]

    print('EndmemberMatrix: ', endmemberMatrix)
    #print('Chromophores: ', chromophores)
    # returns the peusdo inverse of the absorption (endmember) matrix and the str list of chromophores.
    return linalg.pinv(endmemberMatrix)


def flupai(pseudo_inverse_absorption_matrix, data):
    """Fast Linear Unmixing for PhotoAcoustic Imaging (FLUPAI) is based on SVD decomposition with a pseudo inverse,
     which is equivalent to a least squares ansatz for linear spectral unmixing of multi-spectral photoacoustic images.

    :param pseudo_inverse_absorption_matrix: Pseudo inverse, which is calculated by inverting the absorption information
    matrix (number chromophores; number of wavelengths).
    :type pseudo_inverse_absorption_matrix: numpy.ndarray
    :param data: Image data, where the XY-slice corresponds to a single wavelength PA image and the Z-dimension
    corresponds to images of different wavelength.
    :type data: numpy.array
    :return: list with unmixed images containing the chromophore concentration.
    :raise: SystemExit.
    """
    # reshape image data to [number of wavelength, number of pixel]
    # swap x and z axis (x,y,z--> z,y,x) because z contains the wavelength information
    try:
        dims_raw = np.shape(data)
        print("dims_raw:", dims_raw)
        reshapedData = np.reshape(data, (dims_raw[0], -1))
        dims_post = np.shape(reshapedData)
        print("dims_post", dims_post)
    except:
        print('FLUPAI failed probably caused by wrong input dimensions of the input data.')
        print('Input dimension: ', dims_raw)
        print('FLUPAI expects a 3 dimensional numpy array, where first and second dimension are'
              'representing a single wavelength PA image, and the third dimension represents the wavelengths.')
        sys.exit('Reshaping input data failed!!!')

    # matmul of x = PI * b with x chromophore information, PI pseudo inverse with absorber information and b
    # containing the measured pixel.
    try:
        output = np.matmul(pseudo_inverse_absorption_matrix, reshapedData)
    except Exception as e:
        print('Matrix multiplication "PI * b" failed. Pseudo inverse (PI) and input data (b) have matching sizes.')
        print('Input data z-dimension: ', dims_raw[0])
        print('Pseudo Inverse number of wavelengths: ', len(pseudo_inverse_absorption_matrix[0]))
        print(e)
        sys.exit('Matrix multiplication failed!!!')

    # write output into list of images containing the chromophore information.
    numberChromophores = len(pseudo_inverse_absorption_matrix)
    chromophores = []
    for chromophore in range(numberChromophores):
        # swap "x" and "y" axis (y,x --> x,y) to undo previous change
        chromophores.append(np.reshape(output[chromophore, :], (dims_raw[1], dims_raw[2], dims_raw[3])))
    return chromophores
