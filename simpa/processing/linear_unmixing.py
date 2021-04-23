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
        wavelength = self.global_settings[Tags.WAVELENGTH]
        #data_array = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        # testing
        #print('Shape data_array: ', np.shape(data_array))

        sys.exit('Copy of noise processing. +++ WIP +++ ')
        meaningfull_name = 'test'
        #save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], meaningfull_name,)

        self.logger.info("Applying Gaussian Noise Model...[Done]")


# +++++ COPIED FROM CMD TOOLS


def create_piv_absorption_matrix(xmlFile, spec='iTheraSpectrum.'):
    """ Method that returns a str list of choosen chromophores, which are read out of a xml file as well as the
    pseudo inverse of the  absorption (endmember) matrix.

    :param xmlFile: name.options.xml containing the spectral unmxiing settings
    :type xmlFile: str
    :param spec: name of spectrum file as pandas data frame pickle
    :type spec: str
    :return: pseudo inverse, str. list of chromophores in 'correct' order
    """
    # read spectrum and xml file
    settings = su_xml_reader(xmlFile)
    try:
        spectrum = pd.read_pickle('gratzerSpectrum.pkl')
    except Exception as e:
        print(e)
        print('Spectrum not found at the script path.')
        sys.exit(1)
    wavelengths = np.array(get_wavelengths_ordered(xmlFile).split(' ')).astype(int)
    chromophores = []
    chromoWaves = []
    numberWavelengths = len(wavelengths)
    print('Wavelengths: ', wavelengths)

    # search for valid chromophores and append a name and wavelength list for each chromophore
    for key, value in zip(settings.keys(), settings.values()):
        if type(value) != bool:
            chromophores.append(key)
            chromoWaves.append(np.array(value).astype(int))

    numberChromophores = len(chromophores)
    endmemberMatrix = np.zeros((numberWavelengths, numberChromophores))

    # write absorption data fpr each chromophore and the corresponding wavelength into an array (matrix)
    for chrom in range(numberChromophores):
        # cw is a counter to ensure that not selected wavelengths of a chromophore lead to a 0 as entry in the
        # absorption matrix. Cw therefor can be seen as matching number between all wavelengths and the selected ones.
        cw = 0
        for wave in range(numberWavelengths):
            if wavelengths[wave] == chromoWaves[chrom][cw]:
                endmemberMatrix[wave][chrom] = spectrum[chromophores[chrom]][chromoWaves[chrom][cw]]
                cw += 1
            if cw == len(chromoWaves[chrom]):
                break
    #print('EndmemberMatrix: ', endmemberMatrix)
    #print('Chromophores: ', chromophores)
    # returns the peusdo inverse of the absorption (endmember) matrix and the str list of chromophores.
    return linalg.pinv(endmemberMatrix), chromophores


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
    dims = np.shape(data)
    try:
        reshapedData = np.reshape(np.swapaxes(data, axis1=0, axis2=2), (dims[2], -1))
    except:
        print('FLUPAI failed probably caused by wrong input dimensions of the input data.')
        print('Input dimension: ', dims)
        print('FLUPAI expects a 3 dimensional numpy array, where first and second dimension are'
              'representing a single wavelength PA image, and the third dimension represents the wavelengths.')
        sys.exit('Reshaping input data failed!!!')

    # matmul of x = PI * b with x chromophore information, PI pseudo inverse with absorber information and b
    # containing the measured pixel.
    try:
        output = np.matmul(pseudo_inverse_absorption_matrix, reshapedData)
    except Exception as e:
        print('Matrix multiplication "PI * b" failed. Pseudo inverse (PI) and input data (b) have matching sizes.')
        print('Input data z-dimension: ', dims(2))
        print('Pseudo Inverse number of wavelengths: ', len(pseudo_inverse_absorption_matrix[0]))
        print(e)
        sys.exit('Matrix multiplication failed!!!')

    # write output into list of images containing the chromophore information.
    numberChromophores = len(pseudo_inverse_absorption_matrix)
    chromophores = []
    for chromophore in range(numberChromophores):
        # swap "x" and "y" axis (y,x --> x,y) to undo previous change
        chromophores.append(np.swapaxes(np.reshape(output[chromophore, :], (dims[1], dims[0], 1)), axis1=0, axis2=1))
    return chromophores
