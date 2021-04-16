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


import numpy as np
import os
from scipy.ndimage import zoom
from skimage.restoration import estimate_sigma
import time
from simpa.utils import Tags
from simpa.utils.libraries.literature_values import OpticalTissueProperties, StandardProperties
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
from simpa.core.optical_simulation.mcx_adapter import McxAdapter
from simpa.utils.settings_generator import Settings
from simpa.io_handling import load_hdf5
from simpa.utils import TISSUE_LIBRARY
from simpa.core.simulation_components import ProcessingComponent


def preprocessing(self, image_data, scattering, global_settings, component_settings, optical_settings):
    """
    Preprocesses image data and scattering distribution for iterative algorithm using mcx.

    :param image_data: Raw input image of initial pressure.
    :type image_data: numpy array
    :param scattering: Map of scattering coefficients known a priori.
    :type scattering: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :return: Two 3-d numpy arrays and boolean statement: Resampled and (if necessary) stacked volume of noisy
             initial pressure, scattering and bool indicating if image had to be stacked to 3-d.
    """

    if len(np.shape(image_data)) == 2:
        stacking_to_volume = True
        image_data, scattering= stacking(image_data, scattering, global_settings)
    else:
        stacking_to_volume = False

    image_data, scattering = downscaling(self, image_data, scattering,
                                         global_settings, component_settings, optical_settings)
    image_data = add_gaussian_noise(self, image_data, component_settings)

    return image_data, scattering, stacking_to_volume


def stacking(image_data, scattering, global_settings):
    """
    Stacks the image and scattering map in sequence along axis=1 to build 3-d volumes.

    :param image_data: 2-d image data of initial pressure.
    :type image_data: numpy array
    :param scattering: 2-d map of scattering coefficients.
    :type scattering: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :return: Two 3-d numpy arrays: Volumes of stacked input image and scattering.
    """
    num_repeats = 60

    if (Tags.DIM_VOLUME_X_MM in global_settings) and (Tags.DIM_VOLUME_Y_MM in global_settings):
        spacing = global_settings[Tags.DIM_VOLUME_X_MM] / np.shape(image_data)[0]
        num_repeats = int(global_settings[Tags.DIM_VOLUME_Y_MM] / spacing)

    stacked_image = np.stack([image_data] * num_repeats, axis=1)
    stacked_scattering = np.stack([scattering] * num_repeats, axis=1)

    return stacked_image, stacked_scattering


def downscaling(self, image_data, scattering, global_settings, component_settings, optical_settings):
    """
    Downscales the input image and scattering map by a given scale.

    :param image_data: Raw input image of initial pressure.
    :type image_data: numpy array
    :param scattering: Map of scattering coefficients.
    :type scattering: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :return: Two numpy arrays: Downscaled image and scattering map.
    """

    downscaling_method = "nearest"
    scale = 0.73

    if Tags.DOWNSCALE_FACTOR in component_settings:
        scale = float(component_settings[Tags.DOWNSCALE_FACTOR])

    self.logger.debug(f"Downscale factor: {scale}")

    downscaled_image = zoom(image_data, scale, order=1, mode=downscaling_method)
    downscaled_scattering = zoom(scattering, scale, order=1, mode=downscaling_method)

    new_spacing = global_settings[Tags.SPACING_MM] / scale

    if global_settings[Tags.SPACING_MM] != new_spacing:
        global_settings[Tags.SPACING_MM] = new_spacing

    if Tags.ILLUMINATION_POSITION in optical_settings:
        pos = [int(elem * scale) for elem in optical_settings[Tags.ILLUMINATION_POSITION]]
        optical_settings[Tags.ILLUMINATION_POSITION] = pos
    if Tags.ILLUMINATION_DIRECTION in optical_settings:
        optical_settings[Tags.ILLUMINATION_DIRECTION] = list(map(int, optical_settings[Tags.ILLUMINATION_DIRECTION]))
    if Tags.ILLUMINATION_PARAM1 in optical_settings:
        param1 = [int(elem * scale) for elem in optical_settings[Tags.ILLUMINATION_PARAM1]]
        optical_settings[Tags.ILLUMINATION_PARAM1] = param1
    if Tags.ILLUMINATION_PARAM1 in optical_settings:
        param2 = [int(elem * scale) for elem in optical_settings[Tags.ILLUMINATION_PARAM2]]
        optical_settings[Tags.ILLUMINATION_PARAM2] = param2

    return downscaled_image, downscaled_scattering


def add_gaussian_noise(self, image_data, component_settings):
    """
    Adds defined Gaussian model to the input data to simulate noise.

    :param image_data: Image data without noise.
    :type image_data: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :return: Numpy array of the same shape as the input data.
    """
    mean_noise = 0
    std_noise = 0.4

    if Tags.NOISE_STD in component_settings:
        std_noise = float(component_settings[Tags.NOISE_STD])

    self.logger.debug(f"Noise std: {std_noise}")

    image_data = image_data + np.random.normal(mean_noise, std_noise, size=np.shape(image_data))
    image_data[image_data <= 0] = 1e-16

    return image_data


def forward_model_fluence(absorption, scattering, anisotropy, global_settings, optical_settings):
    """
    Simulates photon propagation in 3-d volume and returns simulated fluence map in units of J/cm^2.

    :param absorption: Volume of absorption coefficients in 1/cm for Monte Carlo Simulation.
    :type absorption: numpy array
    :param scattering: Volume of scattering coefficients in 1/cm for Monte Carlo Simulation.
    :type scattering: numpy array
    :param anisotropy: Volume of anisotropy data for Monte Carlo Simulation.
    :type anisotropy: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :return: Numpy array of same shape as input arrays: Fluence map.
    :raises: AssertionError: if Tags.OPTICAL_MODEL tag was not or incorrectly defined in settings.
    """

    if Tags.OPTICAL_MODEL not in optical_settings:
        raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings.")
    model = optical_settings[Tags.OPTICAL_MODEL]

    if model == Tags.OPTICAL_MODEL_MCX:
        forward_model_implementation = McxAdapter(global_settings)
    else:
        raise AssertionError("Tags.OPTICAL_MODEL tag must be Tags.OPTICAL_MODEL_MCX.")

    fluence = forward_model_implementation.forward_model(absorption_cm=absorption,
                                                         scattering_cm=scattering,
                                                         anisotropy=anisotropy)

    print("Simulating the optical forward process...[Done]")

    return fluence


def reconstruct_absorption(image_data, fluence, global_settings, optical_settings, sigma):
    """
    Reconstructs map of absorption coefficients in 1/cm given measured data and simulated fluence.

    :param image_data: Measured image data (initial pressure) used for reconstruction.
    :type image_data: numpy array
    :param fluence: Simulated fluence map in J/cm^2.
    :type fluence: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :param sigma: Regularization factor to avoid instability if the fluence is low.
    :type sigma: numpy array or float
    :return: Numpy array with the same shape as input array: reconstructed absorption.
    """

    if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in optical_settings:
        if Tags.PROPERTY_GRUNEISEN_PARAMETER in global_settings:
            gamma = global_settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
        else:
            gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
            gamma = gamma * np.ones(np.shape(image_data))
        denominator = (fluence + sigma) * gamma * (optical_settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
        absorption = np.array(image_data / denominator)
    else:
        absorption = image_data / (fluence + sigma)

    return absorption


def log_sum_squared_error(image_data, absorption, fluence, global_settings, optical_settings, sigma):
    """
    Computes log (base 10) of the sum of squared error between image and reconstructed pressure map in middle slice.

    :param image_data: Measured image data used for reconstruction.
    :type image_data: numpy array
    :param absorption: Reconstructed map of absorption coefficients in 1/cm.
    :type absorption: numpy array
    :param fluence: Simulated fluence map in J/cm^2.
    :type fluence: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :param sigma: Regularization parameter to avoid instability if the fluence is low.
    :type sigma: numpy array or float
    :return: Float: error.
    """

    if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in optical_settings:
        if Tags.PROPERTY_GRUNEISEN_PARAMETER in global_settings:
            gamma = global_settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
        else:
            gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
            gamma = gamma * np.ones(np.shape(image_data))
        correction_factor = (optical_settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
        predicted_pressure = absorption * (fluence + sigma) * gamma * correction_factor
    else:
        predicted_pressure = absorption * (fluence + sigma)

    y_pos = int(image_data.shape[1] / 2)
    sse = np.sum(np.square(image_data[:, y_pos, :] - predicted_pressure[:, y_pos, :]))

    return np.log10(sse)


def regularization_sigma(self, input_image, component_settings):
    """
    Computes spatial (same shape as input image) or constant regularization parameter sigma.

    :param input_image: Noisy input image.
    :type input_image: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :return: Numpy array or float: regularization parameter.
    :raises: ValueError: if estimated noise is zero, so SNR cannot be computed.
    """
    sigma = 1.0
    noise = float(estimate_sigma(input_image))

    if noise == 0.0:
        raise ValueError("An estimated noise level of zero cannot be used to compute a signal to noise ratio.")

    signal_noise_ratio = input_image / noise

    if Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION in component_settings:
        if component_settings[Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION]:
            sigma = 1e-2
            if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in component_settings:
                sigma = component_settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]
        self.logger.info("Regularization: constant")
        self.logger.debug(f"Sigma: {sigma}")
    else:
        self.logger.info("Regularization: SNR (spatially) dependent")
        sigma = 1 / signal_noise_ratio
        sigma[sigma > 1e8] = 1e8
        sigma[sigma < 1e-8] = 1e-8

    return sigma


def optical_properties(image_data, global_settings):
    """
    Returns a optical properties dictionary containing scattering coefficients and anisotropy.

    :param image_data: Measured image data.
    :type image_data: numpy array
    :param copied_global_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_global_settings: dict
    :return: Dictionary: Optical properties (scattering and anisotropy).
    """

    shape = np.shape(image_data)

    # scattering must be known a priori at the moment.
    if Tags.PROPERTY_SCATTERING_PER_CM in global_settings:
        mus = float(global_settings[Tags.PROPERTY_SCATTERING_PER_CM]) * np.ones(shape)
    else:
        background_dict = TISSUE_LIBRARY.muscle()
        mus = float(MolecularComposition.get_properties_for_wavelength(background_dict, wavelength=800)["mus"])
        mus = mus * np.ones(shape)

    if Tags.PROPERTY_ANISOTROPY in global_settings:
        g = float(global_settings[Tags.PROPERTY_ANISOTROPY]) * np.ones(shape)
        g[:, :, :] = 0.9  # anisotropy is always 0.9 when simulating
    else:
        g = float(OpticalTissueProperties.STANDARD_ANISOTROPY) * np.ones(shape)

    optical_properties = {
        "scattering": mus,
        "anisotropy": g
    }

    return optical_properties


def stopping_criterion(errors, component_settings, iteration):
    """
    Serves as a stopping criterion for the iterative algorithm. If False the iterative algorithm continues.

    :param errors: List of log (base 10) sum of squared errors.
    :type errors: list of floats
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :param iteration: Iteration number.
    :type iteration: int
    :return: Boolean statement: If iteration method should be stopped.
    :raises: AssertionError: if Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL tag is zero
    """
    epsilon = 0.03

    if Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL in component_settings:
        if component_settings[Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL] == 0:
            raise AssertionError("Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL should be greater than zero.")
        epsilon = component_settings[Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL]

    if iteration == 0:
        return False
    elif iteration > 0:
        share = np.abs(errors[iteration - 1] - errors[iteration]) / errors[iteration - 1]
        if share <= epsilon:
            return True
        else:
            return False
    else:
        raise ValueError("Iteration number is negative.")


def iterative_method(self, image_data, scattering, global_settings, component_settings, optical_settings):
    """
    Performs quantitative photoacoustic image reconstruction of absorption distribution by use of an iterative method.
    The distribution of scattering coefficients must be known a priori.

    :param image_data: Measured image data (initial pressure distribution), should be 2- or 3-d.
    :type image_data: numpy array
    :param scattering: A priori-known scattering distribution, should have same dimensions as image_data.
    :type scattering: numpy array
    :param copied_settings: SIMPA settings dictionary that contains the simulation instructions.
    :type copied_settings: dict
    :return: Numpy array: reconstructed absorption coefficients in 1/cm.
    :raises: TypeError: if input data are not passed as a certain type
             ValueError: if input data cannot be used for reconstruction due to shape or value
             FileNotFoundError: if paths stored in settings cannot be accessed
             AssertionError: is Tags.MAX_NUMBER_ITERATIVE_RECONSTRUCTION tag is zero
    """

    # checking input data
    if not isinstance(image_data, np.ndarray):
        raise TypeError("Image data is not a numpy ndarray.")
    elif image_data.size == 0:
        raise ValueError("Image data is empty.")
    elif (len(image_data.shape) < 2) or (len(image_data.shape) > 3):
        raise ValueError("Image data is invalid. Data must be two or three dimensional.")

    if not isinstance(scattering, np.ndarray):
        raise TypeError("Scattering input is not a numpy ndarray.")
    elif scattering.shape != image_data.shape:
        raise ValueError("Shape of scattering data is invalid. Scattering must have the same shape as image_data.")

    # preprocessing for mcx_adapter
    image_data, scattering, stacking_to_volume = preprocessing(self, image_data, scattering, global_settings,
                                                               component_settings, optical_settings)

    # get optical properties necessary for simulation
    optical_properties_dict = optical_properties(image_data, global_settings)
    anisotropy = optical_properties_dict["anisotropy"]

    # regularization parameter sigma
    sigma = regularization_sigma(self, image_data, component_settings)

    # initialization
    absorption = 1e-16 * np.ones(np.shape(image_data))
    error_list = []

    nmax = 10
    if Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER in component_settings:
        if component_settings[Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER] == 0:
            raise AssertionError("Tags.MAX_NUMBER_ITERATIVE_RECONSTRUCTION tag is invalid (equals zero).")
        nmax = int(component_settings[Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER])

    # algorithm
    start_time = time.time()

    i = 0
    while i < nmax:
        print("Iteration: ", i)
        fluence = forward_model_fluence(absorption, scattering, anisotropy, global_settings, optical_settings)
        error_list.append(log_sum_squared_error(image_data, absorption, fluence, global_settings, optical_settings, sigma))
        absorption = reconstruct_absorption(image_data, fluence, global_settings, optical_settings, sigma)

        if stopping_criterion(error_list, component_settings, iteration=i):
            break
        i += 1

    print("--- %s seconds/iteration ---" % round((time.time() - start_time) / (i + 1), 2))

    # extracting field of view if necessary
    if stacking_to_volume:
        y_pos = int(absorption.shape[1] / 2)
        absorption = absorption[:, y_pos, :]

    return absorption


class IterativeqPAI(ProcessingComponent):
    """

    """

    def run(self):

        self.logger.info("Reconstructing absorption using iterative qPAI method...")

        copied_global_settings = Settings(self.global_settings)
        copied_component_settings = Settings(self.component_settings)
        copied_optical_settings = Settings(copied_global_settings[Tags.OPTICAL_MODEL_SETTINGS])

        file = load_hdf5(copied_global_settings[Tags.SIMPA_OUTPUT_PATH])
        wavelength = copied_global_settings[Tags.WAVELENGTH]

        # check if simulation_path and optical_model_binary_path exist
        self.logger.debug(f"Simulation path: {copied_global_settings[Tags.SIMULATION_PATH]}")
        self.logger.debug(f"Optical model binary path: {copied_optical_settings[Tags.OPTICAL_MODEL_BINARY_PATH]}")

        if not os.path.exists(copied_global_settings[Tags.SIMULATION_PATH]):
            print("Tags.SIMULATION_PATH tag in settings cannot be found.")

        if not os.path.exists(copied_optical_settings[Tags.OPTICAL_MODEL_BINARY_PATH]):
            print("Tags.OPTICAL_MODEL_BINARY_PATH tag in settings cannot be found.")

        # get initial pressure and scattering
        image = file["simulations"]["optical_forward_model_output"]["initial_pressure"][str(wavelength)]
        mus = file["simulations"]["simulation_properties"]["mus"][str(wavelength)]

        # run reconstruction
        absorption = iterative_method(self=self,
                                      image_data=image,
                                      scattering=mus,
                                      global_settings=copied_global_settings,
                                      component_settings=copied_component_settings,
                                      optical_settings=copied_optical_settings)


        dst = copied_global_settings[Tags.SIMULATION_PATH]
        np.save(dst + "/Reconstructed_absorption_" + copied_global_settings[Tags.VOLUME_NAME] + ".npy", absorption)

        self.logger.info("Reconstructing absorption using iterative qPAI method...[Done]")


import matplotlib.pyplot as plt
abs = np.load("/home/p253n/Patricia/test/Reconstructed_absorption_MyVolumeName_471.npy")

y_pos = int(abs.shape[1] / 2)
plt.imshow(np.rot90(abs[:, y_pos, :], -1))
plt.colorbar()
plt.show()


#def run_iterative_reconstruction(path_to_hdf5, simulation_path=None, optical_model_binary_path=None):
#     """
#     Performs quantitative photoacoustic image reconstruction of absorption distribution by use of an iterative method.
#
#     :param path_to_hdf5: Path to hdf5 file containing SIMPA settings, initial pressure, and scattering distribution.
#     :type path_to_hdf5: str
#     :param simulation_path: Path where meta data of fluence simulation should be stored temporarily.
#     :type simulation_path: str
#     :param optical_model_binary_path: Path where bin/mcx is located.
#     :type optical_model_binary_path: str
#     :return: Numpy array: reconstructed absorption coefficients in 1/cm.
#     """
#
#     # get data
#     data = load_hdf5(path_to_hdf5)
#     copied_settings = Settings(data["settings"])
#     wavelength = copied_settings[Tags.WAVELENGTHS][0]
#
#     # if simulation_path and optical_model_binary_path are passed
#     if simulation_path != None:
#         copied_settings[Tags.SIMULATION_PATH] = simulation_path
#     if optical_model_binary_path != None:
#         copied_settings[Tags.OPTICAL_MODEL_BINARY_PATH] = optical_model_binary_path
#
#     # get initial pressure and scattering
#     if "original_data" in data["simulations"]:
#         image = data["simulations"]["original_data"]["optical_forward_model_output"][str(wavelength)]["initial_pressure"]
#         mus = data["simulations"]["original_data"]["simulation_properties"][str(wavelength)]["mus"]
#     else:
#         image = data["simulations"]["optical_forward_model_output"]["initial_pressure"][str(wavelength)]
#         mus = data["simulations"]["simulation_properties"]["mus"][str(wavelength)]
#
#     # run reconstruction
#     absorption = iterative_method(image_data=image, scattering=mus, copied_settings=copied_settings)
#
#     return absorption






"""
Testing
"""
# import matplotlib.pyplot as plt
#
# PATH = "/home/p253n/Patricia/qPAI_CoxAlgorithm/Testing_for_SIMPA/simple_volume_data/Volume_99/Wavelength_700.hdf5"
# SIM_PATH = "/home/p253n/Patricia/qPAI_CoxAlgorithm/Testing_for_SIMPA"
# OPT_PATH = "/home/p253n/Patricia/msot_mcx/bin/mcx"
#
# absorption = run_iterative_reconstruction(PATH, simulation_path=SIM_PATH, optical_model_binary_path=OPT_PATH)
#
# if len(absorption.shape) == 3:
#     y_pos = int(absorption.shape[1] / 2)
#     absorption = absorption[:, y_pos, :]
#
# plt.imshow(np.rot90(absorption, -1))
# plt.colorbar()
# plt.show()


