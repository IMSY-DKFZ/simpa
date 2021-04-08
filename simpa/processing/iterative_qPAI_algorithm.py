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
from simpa.utils import Tags
from scipy.ndimage import zoom
from skimage.restoration import estimate_sigma
from simpa.utils.libraries.literature_values import OpticalTissueProperties, StandardProperties
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
#from simpa.core.optical_simulation.mcx_adapter import McxAdapter
from simpa.utils.settings_generator import Settings
from simpa.io_handling import load_hdf5
from simpa.utils import TISSUE_LIBRARY
from mcx_adapter import McxAdapter


def preprocess_for_mcx(image_data, settings):
    """
    Preprocesses image data for iterative algorithm using mcx_adapter.

    :param image_data: (numpy array) Image to be preprocessed.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Preprocessed image and flag indicating if image was stacked to 3-d.
    """

    if (settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW] == True) or (len(np.shape(image_data)) == 2):
        stacking_to_volume = True

        print("Input 2-d image is stacked to 3-d.")
        image_data = stacking(image_data, settings)
        settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW] = False
    else:
        stacking_to_volume = False

    image_data = downscaling(image_data, settings)
    image_data = add_gaussian_noise(image_data, settings)

    return image_data, stacking_to_volume


def stacking(image_data, settings):
    """
    Stacks the image in sequence vertically to build 3-d volume.

    :param image_data: (numpy array) Image to be stacked to 3-d volume.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Volume of stacked input image.
    """

    num_repeats = 60

    if (Tags.DIM_VOLUME_X_MM in settings) and (Tags.DIM_VOLUME_Y_MM in settings):
        spacing = settings[Tags.DIM_VOLUME_X_MM] / np.shape(image_data)[0]
        num_repeats = int(settings[Tags.DIM_VOLUME_Y_MM] / spacing)

    stacked_image = np.swapaxes(np.dstack([image_data] * num_repeats), axis1=1, axis2=2)

    return stacked_image


def downscaling(image_data, settings):
    """
    Downscales the input image with a given method by a given scale.

    :param image_data: (numpy array) Image to be downscaled.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Downscaled image.
    """

    downscaling_method = "nearest"
    scale = 0.73

    if Tags.ITERATIVE_RECONSTRUCTION_DOWNSCALING_FACTOR in settings:
        scale = float(settings[Tags.ITERATIVE_RECONSTRUCTION_DOWNSCALING_FACTOR])

    print("Image is resampled with scaling factor: ", scale)
    downscaled_image = zoom(image_data, scale, order=1, mode=downscaling_method)

    new_spacing = settings[Tags.SPACING_MM] / scale
    print("The new voxel spacing (in mm) has the value: ", round(new_spacing, 6))

    return downscaled_image


def add_gaussian_noise(image_data, settings):
    """
    Adds the defined Gaussian noise model to the input data.

    :param image_data: (numpy array) Image data the noise should be added to.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Numpy array of the same shape as the input data.
    """

    mean_noise = 0
    std_noise = 1e-5 * np.max(image_data)

    if Tags.ITERATIVE_RECONSTRUCTION_NOISE_STD in settings:
        std_noise = float(settings[Tags.ITERATIVE_RECONSTRUCTION_NOISE_STD])

    print("Standard deviation of gaussian noise model: ", round(std_noise, 6))

    image_data = image_data + np.random.normal(mean_noise, std_noise, size=np.shape(image_data))
    image_data[image_data <= 0] = 1e-16

    return image_data


def model_fluence(absorption, scattering, anisotropy, settings):
    """
    Simulates photon propagation in 3-d volume and returns simulated fluence map in units of J/cm^2.

    :param absorption: (numpy array) Absorption coefficients [1/cm] for Monte Carlo Simulation.
    :param scattering: (numpy array) Scattering coefficients [1/cm] for Monte Carlo Simulation.
    :param anisotropy: (numpy array) Anisotropy data for Monte Carlo Simulation.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Numpy array of same shape as input arrays.
    """

    if Tags.OPTICAL_MODEL not in settings:
        raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings.")

    model = settings[Tags.OPTICAL_MODEL]
    forward_model_implementation = None

    if model == Tags.OPTICAL_MODEL_MCX:
        forward_model_implementation = McxAdapter()
    else:
        raise AssertionError("Tags.OPTICAL_MODEL tag must be Tags.OPTICAL_MODEL_MCX.")

    spacing = float(settings[Tags.DIM_VOLUME_X_MM] / np.shape(absorption)[0])
    if settings[Tags.SPACING_MM] != spacing:
        settings[Tags.SPACING_MM] = spacing

    fluence = forward_model_implementation.forward_model(absorption_cm=absorption,
                                                         scattering_cm=scattering,
                                                         anisotropy=anisotropy,
                                                         settings=settings,
                                                         scale=1.0)

    print("Simulating the optical forward process...[Done]")

    return fluence


def reconstruct_absorption(image_data, fluence, settings, sigma):
    """
    Reconstructs map of absorption coefficients given measured image and simulated fluence.

    :param image_data: (numpy array) Measured image data used for reconstruction.
    :param fluence: (numpy array) Simulated fluence map.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :param sigma: (numpy array or float) Regularization factor to avoid instability where the fluence is low.
    :return: Numpy array with the same shape as input arrays.
    """

    if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:

        if Tags.PROPERTY_GRUNEISEN_PARAMETER in settings:
            gamma = settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
        else:
            gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
            gamma = gamma * np.ones(np.shape(image_data))

        denominator = (fluence + sigma) * gamma * (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
        absorption = np.array(image_data / denominator)

    else:
        absorption = image_data / (fluence + sigma)

    return absorption


def log_sum_squared_error(image_data, absorption, fluence, settings, sigma):
    """
    Computes log (base 10) of the sum of squared error between volume and reconstructed pressure map.

    :param image_data: (numpy array) Measured image data used for reconstruction.
    :param absorption: (numpy array) Predicted map of absorption coefficients.
    :param fluence: (numpy array) Simulated fluence map.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :param sigma: (numpy array or float) Regularization factor to avoid instability where the fluence is low.
    :return: Float.
    """

    if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:

        if Tags.PROPERTY_GRUNEISEN_PARAMETER in settings:
            gamma = settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
        else:
            gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
            gamma = gamma * np.ones(np.shape(image_data))

        correction_factor = (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
        predicted_pressure = absorption * (fluence + sigma) * gamma * correction_factor

    else:
        predicted_pressure = absorption * (fluence + sigma)

    sse = np.sum(np.square(image_data - predicted_pressure))

    return np.log10(sse)


def regularization_sigma(input_image, settings):
    """
    Computes spatial or constant regularization parameter sigma.

    :param input_image: (numpy array) Noisy image.
    :return: Numpy array or float.
    """

    noise = float(estimate_sigma(input_image))
    print("Estimated standard deviation of gaussian noise model: ", round(noise, 6))

    if noise == 0.0:
        raise ValueError("An estimated noise level of zero cannot be used to compute a signal to noise ratio.")

    signal_noise_ratio = input_image / noise

    if Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION in settings:
        if settings[Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION]:
            sigma = 1e-2

            if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in settings:
                sigma = settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]

            print("Regularization is chosen as constant with a value of: ", sigma)

    else:
        print("Regularization is chosen to be spatially dependent.")
        sigma = 1 / signal_noise_ratio
        sigma[sigma > 1e8] = 1e8
        sigma[sigma < 1e-8] = 1e-8

    return sigma


def optical_properties(image_data, settings):
    """
    Returns scattering coefficients and anisotropy in dictionary of optical properties.

    :param image_data: (numpy array) Measured image data.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Dictionary containing optical properties.
    """

    shape = np.shape(image_data)

    # if Tags.PROPERTY_SCATTERING_PER_CM in settings:
    #     mus = float(settings[Tags.PROPERTY_SCATTERING_PER_CM]) * np.ones(shape)
    # else:
    #     background_dict = TISSUE_LIBRARY.muscle()
    #     mus = float(MolecularComposition.get_properties_for_wavelength(background_dict, wavelength=800)["mus"])
    #     mus = mus * np.ones(shape)

    if Tags.PROPERTY_ANISOTROPY in settings:
        g = float(settings[Tags.PROPERTY_ANISOTROPY]) * np.ones(shape)
    else:
        g = float(OpticalTissueProperties.STANDARD_ANISOTROPY) * np.ones(shape)

    optical_properties = {
        # "scattering": mus,
        "anisotropy": g
    }

    return optical_properties


def stopping_criterion(errors, settings, iteration):
    """
    Serves as a stopping criterion for the iterative algorithm. If False the iterative algorithm continues.

    :param errors: (list of floats) List of log (base 10) sum of squared errors.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :param iteration: (int) Iteration number.
    :return: Bool.
    """
    epsilon = 0.005

    if Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL in settings:

        if settings[Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL] == 0:
            raise AssertionError("Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL should be greater than zero.")

        epsilon = settings[Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL]

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


def run_iterative_reconstruction(image_data, settings, mus):
    """
    Performs quantitative photoacoustic image reconstruction of absorption distribution by use of a iterative method.

    :param image_data: (numpy array) Measured image data (inital pressure) from which is reconstructed.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Numpy array of absorption coefficients in 1/cm.
    """

    # checking input
    if not isinstance(image_data, np.ndarray):
        raise TypeError("Input data is not a numpy ndarray.")
    elif image_data.size == 0:
        raise ValueError("Input data is empty.")
    elif (len(image_data.shape) < 2) or (len(image_data.shape) > 3):
        raise TypeError("Input data is invalid. Data must be two or three dimensional.")

    if not isinstance(settings, Settings):
        raise TypeError("Use a Settings instance from simpa.utils.settings_generator as simulation input.")

    if not os.path.exists(settings[Tags.SIMULATION_PATH]):
        print("Tags.SIMULATION_PATH tag in settings is invalid.")
        settings[Tags.SIMULATION_PATH] = input("Please enter a valid simulation path: ")

        if not os.path.exists(settings[Tags.SIMULATION_PATH]):
            raise FileNotFoundError("Tags.SIMULATION_PATH tag in settings is still invalid.")

    if not os.path.exists(settings[Tags.OPTICAL_MODEL_BINARY_PATH]):
        print("Tags.OPTICAL_MODEL_BINARY_PATH tag in settings is invalid.")
        settings[Tags.OPTICAL_MODEL_BINARY_PATH] = input("Please enter a valid optical model binary path: ")

        if not os.path.exists(settings[Tags.OPTICAL_MODEL_BINARY_PATH]):
            raise FileNotFoundError("Tags.OPTICAL_MODEL_BINARY_PATH tag in settings is still invalid.")

    # preprocessing
    image_data, stacking_to_volume = preprocess_for_mcx(image_data, settings)

    # get optical properties necessary for simulation
    optical_properties_dict = optical_properties(image_data, settings)
    #scattering = optical_properties_dict["scattering"]
    scattering = mus
    anisotropy = optical_properties_dict["anisotropy"]

    # regularization parameter
    sigma = regularization_sigma(image_data, settings)

    # initialization
    absorption = 1e-16 * np.ones(np.shape(image_data))
    error_list = []

    nmax = 10
    if Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER in settings:

        if settings[Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER] == 0:
            raise AssertionError("Tags.MAX_NUMBER_ITERATIVE_RECONSTRUCTION tag is invalid (equals zero).")

        nmax = settings[Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER]

    # algorithm
    print("Start of iterative method...")

    i = 0
    while i < nmax:
        print("ITERATION: ", i)

        fluence = model_fluence(absorption, scattering, anisotropy, settings)
        error_list.append(log_sum_squared_error(image_data, absorption, fluence, settings, sigma))
        print("log (base 10) sum squared error: ", round(error_list[i], 6))
        absorption = reconstruct_absorption(image_data, fluence, settings, sigma)

        if stopping_criterion(error_list, settings, iteration=i):
            break
        i += 1

    print("...End of iterative method.")

    if stacking_to_volume:
        print("Extracting field of view.")
        y_pos = int(np.shape(image_data)[1] / 2)
        absorption = absorption[:, y_pos, :]

    return absorption


"""
Testing
"""

import matplotlib.pyplot as plt
PATH = "/home/p253n/Patricia/qPAI_CoxAlgorithm/qPAI_segmentation_3D/SemanticForearm_2/Wavelength_700.hdf5"

data = load_hdf5(PATH)
settings = Settings(data["settings"])
#image = data["simulations"]["original_data"]["optical_forward_model_output"]["700"]["initial_pressure"]
image = data["simulations"]["optical_forward_model_output"]["initial_pressure"]["700"]
mus = data["simulations"]["simulation_properties"]["mus"]["700"]
mus = zoom(mus, 0.73, order=1, mode="nearest")


reconstructed_absorption = run_iterative_reconstruction(image_data=image, settings=settings, mus=mus)

plt.imshow(np.rot90(reconstructed_absorption[:, 50, :], -1))
plt.colorbar()
plt.show()








