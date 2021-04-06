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
from simpa.utils import Tags
from scipy.ndimage import zoom
from skimage.restoration import estimate_sigma


def preprocess_for_mcx(image_data, settings):
    """
    Preprocesses image data for iterative algorithm using mcx_adapter.

    :param image_data: (numpy array) Image to be preprocessed.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Preprocessed Image.
    """

    if (settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW] == True) or (len(np.shape(image_data)) == 2):
        image_data = stacking(image_data, settings)

    image_data = downscaling(image_data, settings)
    image_data = add_gaussian_noise(image_data, settings)

    return image_data


def stacking(image_data, settings):
    """
    Stacks the image in sequence vertically to build 3-d volume.

    :param image_data: (numpy array) Image to be stacked to 3-d volume.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Volume of stacked input image.
    """

    num_repeats = 60

    if Tags.DIM_VOLUME_X_MM in settings and Tags.DIM_VOLUME_Y_MM in settings:
        spacing = settings[Tags.DIM_VOLUME_X_MM] / np.shape(image_data)[0]
        num_repeats = int(settings[Tags.DIM_VOLUME_Y_MM] / spacing)

    stacked_image = np.swapaxes(np.dstack([image_data] * num_repeats), axis1=1, axis2=2)
    # vstack?

    return stacked_image


def downscaling(image_data, settings):
    """
    Downscales the input image with a given method by a given scale.

    :param image_data: (numpy array) Image to be downscaled.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :return: Downscaled image.
    """

    print("DOWNSCALING IMAGE")
    downscaling_method = "nearest"
    scale = 0.73

    if Tags.DOWNSCALING_METHOD_ITERATIVE_RECONSTRUCTION in settings:
        downscaling_method = settings[Tags.DOWNSCALING_METHOD_ITERATIVE_RECONSTRUCTION]

    if Tags.DOWNSCALE_FACTOR_ITERATIVE_RECONSTRUCTION in settings:
        scale = float(settings[Tags.DOWNSCALE_FACTOR_ITERATIVE_RECONSTRUCTION])

    downscaled_image = zoom(image_data, scale, order=1, mode=downscaling_method)

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

    if Tags.NOISE_MEAN_ITERATIVE_RECONSTRUCTION in settings:
        mean_noise = float(settings[Tags.NOISE_MEAN_ITERATIVE_RECONSTRUCTION])

    if Tags.NOISE_STD_ITERATIVE_RECONSTRUCTION in settings:
        factor = float(settings[Tags.NOISE_STD_ITERATIVE_RECONSTRUCTION][str(settings[Tags.WAVELENGTH])])
        std_noise = factor * np.max(image_data)

    image_data = image_data + np.random.normal(mean_noise, std_noise, size=np.shape(image_data))
    image_data[image_data <= 0] = 1e-16

    return image_data


def model_fluence(absorption, scattering, anisotropy, settings):
    """
    Simulates photon propagation in 3-d volume and returns simulated fluence map in units of J/m^3.

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
        raise AssertionError("Tags.OPTICAL_MODEL tag is not OPTICAL_MODEL_MCX.")

    original_spacing = settings[Tags.SPACING_MM]

    if Tags.DIM_VOLUME_X_MM in settings:
        spacing = settings[Tags.DIM_VOLUME_X_MM] / np.shape(absorption)[0]

        if settings[Tags.SPACING_MM] != spacing:
            settings[Tags.SPACING_MM] = spacing
    else:
        raise AssertionError("Tags.DIM_VOLUME_X_MM tag was not specified in the settings.")

    fluence = forward_model_implementation.forward_model(absorption_cm=absorption,
                                                         scattering_cm=scattering,
                                                         anisotropy=anisotropy,
                                                         settings=settings)

    print("Simulating the optical forward process...[Done]")
    settings[Tags.SPACING_MM] = original_spacing

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
        gamma = settings[Tags.PROPERTY_GRUNEISEN_PARAMETER]
        correction = 1e6

        denominator = (fluence + sigma) * gamma * (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * correction
        absorption = np.array(image_data / denominator)
    else:
        absorption = image_data / (fluence + sigma)

    return absorption


def log_sum_squared_error(image_data, absorption, fluence, settings, sigma):
    """
    Computes log (base 10) of the sum of squared error between middle slice of image and reconstruction.

    :param image_data: (numpy array) Measured image data used for reconstruction.
    :param absorption: (numpy array) Predicted map of absorption coefficients.
    :param fluence: (numpy array) Simulated fluence map.
    :param settings: (dict) Settings dictionary that contains the simulation instructions.
    :param sigma: (numpy array or float) Regularization factor to avoid instability where the fluence is low.
    :return: Float.
    """

    if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
        gamma = settings[Tags.PROPERTY_GRUNEISEN_PARAMETER]

        correction_factor = (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
        predicted_pressure = absorption * (fluence + sigma) * gamma * correction_factor
    else:
        predicted_pressure = pabsorption * (fluence + sigma)

    y_pos = int(predicted_pressure.shape[1] / 2)
    sse = np.sum(np.square(image_data[:, y_pos, :] - predicted_pressure[:, y_pos, :]))

    return np.log10(sse)


def regularization_sigma(input_image):
    """
    Computes spatial or constant regularization parameter sigma.

    :param input_image: (numpy array) Noisy image.
    :return: Numpy array or float.
    """

    noise = estimate_sigma(input_image)
    print("Estimated noise level: $\simga$ = ", noise)

    signal_noise_ratio = input_image / noise

    if settings[Tags.CONSTANT_REGULARIZATION_ITERATIVE_ALGORITHM]:
        print("CONSTANT REGULARIZATION")

        if Tags.CONSTANT_REGULARIZATION_ITERATIVE_ALGORITHM_SIGMA in settings:
            sigma = Tags.CONSTANT_REGULARIZATION_ITERATIVE_ALGORITHM_SIGMA
        else:
            sigma = 1e-2
    else:
        sigma = 1 / signal_noise_ratio
        sigma[sigma > 1e8] = 1e8
        sigma[sigma < 1e-8] = 1e-8

    return sigma


def scattering(settings):
    return None



# import numpy as np
# import os
# import fnmatch
# from scipy.ndimage import zoom
# from simpa.io_handling import load_hdf5
# from simpa.utils.settings_generator import Settings
# from simpa.utils import Tags
# from mcx_adapter import McxAdapter
# from skimage.restoration import estimate_sigma
#

# """
# set directories/path to data
# """
# PATH_TO_DATA = "/home/p253n/Patricia/qPAI_CoxAlgorithm/UNet_qPAI/2D_GAN/Cox_algorithm/raw_data/"
# SAVE_PATH = "/home/p253n/Patricia/qPAI_CoxAlgorithm/UNet_qPAI/2D_GAN/Cox_algorithm/results/"
# SIMULATION_PATH = "/home/p253n/Patricia/qPAI_CoxAlgorithm/UNet_qPAI/2D_GAN/Cox_algorithm/results"
# MCX_BINARY_PATH = "/home/p253n/Patricia/msot_mcx/bin/mcx"
#
# NOISE = True
# CONSTANT_REGULARIZATION = False
# NUMBER_OF_ITERATIONS = 10
# # TARGET_SPACING = 0.15625
# TARGET_SPACING = 0.2
#
# pattern = "*.hdf5"
# print("start")
#
# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)
#
# detailed_base_dir = dict()
# subfiles = []
# for root, dirs, files in os.walk(PATH_TO_DATA):
#     for file in sorted(files):
#         if (root.split('/')[-2].startswith('raw')) & (root.split('/')[-1].startswith('SemanticForearm')):
#             subfiles.append(os.path.join(root, file))
#     if subfiles:
#         detailed_base_dir[root] = sorted(fnmatch.filter(subfiles, pattern))
#         subfiles = []
#
# for i, data_id in enumerate(detailed_base_dir):
#     print(data_id)
#     for wavelength_idx, name_wavelength in enumerate(detailed_base_dir[data_id]):
#         numpy_array_dict = load_hdf5(detailed_base_dir[data_id][wavelength_idx])
#         print(name_wavelength.split('/')[-1])
#         wavelength = name_wavelength.split('/')[-1][-8:-5]
#         settings = Settings(numpy_array_dict["settings"])
#         settings[Tags.SIMULATION_PATH] = SIMULATION_PATH
#         settings[Tags.OPTICAL_MODEL_BINARY_PATH] = MCX_BINARY_PATH
#
#         """
#         preprocessing
#         """
#         SPACING = settings["voxel_spacing_mm"]
#         if SPACING == TARGET_SPACING:
#             print("no rescaling needed!")
#             RESCALING = False
#             scale = 1.0
#         else:
#             print("rescaling needed!")
#             RESCALING = True
#             scale = SPACING / TARGET_SPACING
#             print("scale: ", scale)
#
#         ABSORPTION = numpy_array_dict["simulations"]["original_data"]["simulation_properties"][wavelength]["mua"]
#         FLUENCE = numpy_array_dict["simulations"]["original_data"]["optical_forward_model_output"][wavelength]["fluence"]
#         initial_pressure = numpy_array_dict["simulations"]["original_data"]["optical_forward_model_output"][wavelength]["initial_pressure"]
#         scattering = numpy_array_dict["simulations"]["original_data"]["simulation_properties"][wavelength]["mus"]
#         anisotropy = numpy_array_dict["simulations"]["original_data"]["simulation_properties"][wavelength]["g"]
#         gamma = numpy_array_dict["simulations"]["original_data"]["simulation_properties"][wavelength]["gamma"]
#
#         # ABSORPTION = numpy_array_dict["simulations"]["simulation_properties"]["mua"][wavelength]
#         # FLUENCE = numpy_array_dict["simulations"]["optical_forward_model_output"]["fluence"][wavelength]
#         # initial_pressure = numpy_array_dict["simulations"]["optical_forward_model_output"]["initial_pressure"][wavelength]
#         # scattering = numpy_array_dict["simulations"]["simulation_properties"]["mus"][wavelength]
#         # anisotropy = numpy_array_dict["simulations"]["simulation_properties"]["g"][wavelength]
#         # gamma = numpy_array_dict["simulations"]["simulation_properties"]["gamma"]
#
#         if (settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW] == True) or (len(ABSORPTION.shape) == 2):
#             settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW] = False
#             num_repeats = int(settings[Tags.DIM_VOLUME_Y_MM] / SPACING)
#             print("2D data is spread out to 3D with repeating factor: ", num_repeats)
#
#             ABSORPTION = spread_to_3D(ABSORPTION, num_repeats=num_repeats)
#             FLUENCE = spread_to_3D(FLUENCE, num_repeats=num_repeats)
#             initial_pressure = spread_to_3D(initial_pressure, num_repeats=num_repeats)
#             scattering = spread_to_3D(scattering, num_repeats=num_repeats)
#             anisotropy = spread_to_3D(anisotropy, num_repeats=num_repeats)
#             gamma = spread_to_3D(gamma, num_repeats=num_repeats)
#
#         anisotropy[:, :, :] = 0.9
#
#         if RESCALING:
#             print("rescaling!")
#             ABSORPTION = resample(ABSORPTION, scale=scale)
#             FLUENCE = resample(FLUENCE, scale=scale)
#             initial_pressure = resample(initial_pressure, scale=scale)
#             scattering = resample(scattering, scale=scale)
#             anisotropy = resample(anisotropy, scale=scale)
#             gamma = resample(gamma, scale=scale)
#
#         size_x, size_y, size_z = ABSORPTION.shape
#         print("volume dimensions: (" + str(size_x) + ", " + str(size_y) + ", " + str(size_z) + ")")
#
#         if NOISE:
#             print("noise is added!")
#             std = 0.4  #np.max(initial_pressure) * 3e-6
#             initial_pressure = add_gaussian_noise(initial_pressure, mean=0.0, std=std)
#             noise = estimate_sigma(initial_pressure)  # noise estimation for spatial dependent regularization
#             print("Estimated noise level = ", noise)
#             SNR = initial_pressure / noise
#
#         """
#         iterative algorithm published by Cox et al.
#         """
#         # Initialization
#         image_data = initial_pressure
#         errors = np.empty([NUMBER_OF_ITERATIONS])
#
#         if NOISE:
#             if CONSTANT_REGULARIZATION:
#                 sigma = 0.01
#                 print("constant sigma = ", sigma)
#             else:
#                 sigma = 1 / SNR  # Regularization factor sigma - to be chosen proportional to the noise level in the image
#                 sigma[sigma > 1e8] = 1e8
#                 sigma[sigma < 1e-8] = 1e-8
#                 print("spatial dependent sigma (min, mean, max) = (" + str(np.min(sigma)) + ", " + str(np.mean(sigma)) + ", " + str(np.max(sigma)) + ")")
#         else:
#             sigma = 0.001
#             print("constant sigma = ", sigma)
#
#         # Iteration
#         absorption = 1e-16 * np.ones(initial_pressure.shape)
#         i = 0
#         while i < NUMBER_OF_ITERATIONS:
#             print("iteration: ", i)
#             fluence = np.array(apply_optical_forward_model(absorption=absorption, scattering=scattering, anisotropy=anisotropy, settings=settings, scale=scale))
#             error = calculate_log10_error(initial_pressure=initial_pressure, absorption=absorption, fluence=fluence, gamma=gamma, sigma=sigma, settings=settings)
#             errors[i] = error
#             print("error = ", error)
#             absorption = calculate_absorption(initial_pressure=initial_pressure, fluence=fluence, gamma=gamma, sigma=sigma, settings=settings)
#             i += 1
#
#         result_array = np.empty([2, size_x, size_y, size_z])
#         result_array[0], result_array[1] = absorption, fluence
#
#         target_array = np.empty([2, size_x, size_y, size_z])
#         target_array[0], target_array[1] = ABSORPTION, FLUENCE
#
#         """
#         saving
#         """
#         OUTPUT_PATH = SAVE_PATH + data_id.split('/')[-1] + "/" + "Wavelength_" + str(wavelength) + "/"
#         if not os.path.exists(OUTPUT_PATH):
#             os.makedirs(OUTPUT_PATH)
#
#         print("SAVING!")
#         dst_image = os.path.join(OUTPUT_PATH, "image.npy")
#         np.save(dst_image, image_data)
#         dst_target = os.path.join(OUTPUT_PATH, "target.npy")
#         np.save(dst_target, target_array)
#         dst_result = os.path.join(OUTPUT_PATH, "result.npy")
#         np.save(dst_result, result_array)
#         dst_error = os.path.join(OUTPUT_PATH, "error.npy")
#         np.save(dst_error, np.array(errors))
#
# print("end")