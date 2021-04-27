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
from typing import Tuple
from simpa.utils import Tags
from simpa.utils.libraries.literature_values import OpticalTissueProperties, StandardProperties
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
from simpa.simulation_components import OpticalForwardModelMcxAdapter
from simpa.utils import Settings
from simpa.io_handling import load_hdf5
from simpa.utils import TISSUE_LIBRARY
from simpa.core.simulation_components import ProcessingComponent


class IterativeqPAIProcessingComponent(ProcessingComponent):
    """
        Applies iterative qPAI Algorithm [1] on simulated initial pressure map and saves the
        reconstruction results in numpy files. If a 2-d map of initial_pressure is passed the algorithm saves
        the reconstructed absorption coefficients as a 2-d map, else a 3-d absorption reconstruction is
        saved.
        The SAVE_PATH entry in path_config.env specifies the location the numpy files are saved at.
        To run the reconstruction the scattering coefficients must be known a priori.
        :param kwargs:
           **Tags.DOWNSCALE_FACTOR (default: 0.73)
           **Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION (default: False)
           **Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER (default: 10)
           **Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA (default: 0.01)
           **Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS (default: False)
           **Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL (default: 0.03)
           **settings (required)
           **component_settings_key (required)

        [1] B. T. Cox et al. 2006, "Two-dimensional quantitative photoacoustic image reconstruction of absorption
        distributions in scattering media by use of a simple iterative method", https://doi.org/10.1364/ao.45.001866
    """

    def __init__(self, global_settings, component_settings_key: str):
        super(ProcessingComponent, self).__init__(global_settings=global_settings)

        self.global_settings = global_settings
        self.optical_settings = global_settings.get_optical_settings()
        self.iterative_method_settings = Settings(global_settings[component_settings_key])

    def run(self):
        self.logger.info("Reconstructing absorption using iterative qPAI method...")

        # check if simulation_path and optical_model_binary_path exist
        self.logger.debug(f"Simulation path: {self.global_settings[Tags.SIMULATION_PATH]}")
        self.logger.debug(f"Optical model binary path: {self.optical_settings[Tags.OPTICAL_MODEL_BINARY_PATH]}")

        if not os.path.exists(self.global_settings[Tags.SIMULATION_PATH]):
            print("Tags.SIMULATION_PATH tag in settings cannot be found.")

        if not os.path.exists(self.optical_settings[Tags.OPTICAL_MODEL_BINARY_PATH]):
            print("Tags.OPTICAL_MODEL_BINARY_PATH tag in settings cannot be found.")

        # debug reconstruction settings
        if Tags.DOWNSCALE_FACTOR in self.iterative_method_settings:
            self.logger.debug(f"Resampling factor: {self.iterative_method_settings[Tags.DOWNSCALE_FACTOR]}")
        else:
            self.logger.debug("Resampling factor: 0.73")

        if Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION in self.iterative_method_settings:
            if self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION]:
                self.logger.debug("Regularization: constant")
                if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in self.iterative_method_settings:
                    self.logger.debug(f"Regularization parameter:"
                            f" {self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]}")
                else:
                    self.logger.debug("Regularization parameter: 0.01")
            else:
                self.logger.debug("Regularization: SNR/spatially dependent")
        else:
            self.logger.debug("Regularization: SNR/spatially dependent")

        if Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER in self.iterative_method_settings:
            self.logger.debug(f"Maximum number of iterations:"
                              f" {self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER]}")
        else:
            self.logger.debug("Maximum number of iterations: 10")

        if Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS in self.iterative_method_settings:
            self.logger.debug(f"Save intermediate absorptions:"
                    f" {self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS]}")
        else:
            self.logger.debug("Save intermediate absorptions: False")

        # run reconstruction
        reconstructed_absorption, list_of_intermediate_absorptions = self.iterative_absorption_reconstruction()

        # save absorption update of last iteration step
        dst = self.global_settings[Tags.SIMULATION_PATH] + "/Reconstructed_qPAI_absorption_"
        np.save(dst + self.global_settings[Tags.VOLUME_NAME] + ".npy", reconstructed_absorption)

        # save a list of all intermediate absorption (2-d only) updates if intended (e.g. for testing algorithm)
        if Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS in self.iterative_method_settings:
            if self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS]:
                dst = self.global_settings[Tags.SIMULATION_PATH] + "/List_reconstructed_qPAI_absorptions_"
                np.save(dst + self.global_settings[Tags.VOLUME_NAME] + ".npy", list_of_intermediate_absorptions)

        self.logger.info("Reconstructing absorption using iterative qPAI method...[Done]")

    def iterative_absorption_reconstruction(self) -> Tuple[np.ndarray, list]:
        """
        Performs quantitative photoacoustic image reconstruction of absorption distribution by use of an
        iterative method. The distribution of scattering coefficients must be known a priori.

        :return: Reconstructed absorption coefficients in 1/cm.
        :raises: TypeError: if input data are not passed as a certain type
                 ValueError: if input data cannot be used for reconstruction due to shape or value
                 FileNotFoundError: if paths stored in settings cannot be accessed
                 AssertionError: is Tags.MAX_NUMBER_ITERATIVE_RECONSTRUCTION tag is zero
        """

        # extract "measured" initial pressure and a priori scattering data
        image_data, scattering = self.extract_initial_data_from_hdf5()

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

        # preprocessing for iterative qPAI method and mcx_adapter
        image_data, scattering, stacked_to_volume = self.preprocessing_for_iterative_qPAI(image_data=image_data,
                                                                                           scattering=scattering)

        # get optical properties necessary for simulation
        optical_properties_dict = self.standard_optical_properties(image_data)
        anisotropy = optical_properties_dict["anisotropy"]

        # regularization parameter sigma
        sigma = self.regularization_sigma(image_data)

        # initialization
        absorption = 1e-16 * np.ones(np.shape(image_data))
        y_pos = int(np.shape(absorption)[1] / 2)  # to extract middle slice
        list_of_intermediate_absorptions = []  # if intentional all intermediate iteration updates can be returned
        error_list = []

        nmax = 10
        if Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER in self.iterative_method_settings:
            if self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER] == 0:
                raise AssertionError("Tags.MAX_NUMBER_ITERATIVE_RECONSTRUCTION tag is invalid (equals zero).")
            nmax = int(self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER])

        # run algorithm
        start_time = time.time()

        i = 0
        while i < nmax:
            print("Iteration: ", i)
            # core method
            fluence = self.forward_model_fluence(absorption, scattering, anisotropy)
            error_list.append(self.log_sum_squared_error(image_data, absorption, fluence, sigma))
            absorption = self.update_absorption_estimate(image_data, fluence, sigma)

            # only store middle slice (2-d image instead of 3-d volume) in iteration list for better performance
            list_of_intermediate_absorptions.append(absorption[:, y_pos, :])

            # check if current error did not change significantly in comparison to preceding error
            if self.convergence_stopping_criterion(error_list, iteration=i):
                break
            i += 1

        print("--- %s seconds/iteration ---" % round((time.time() - start_time) / (i + 1), 2))

        # extracting field of view if input initial pressure was passed as a 2-d array
        if stacked_to_volume:
            absorption = absorption[:, y_pos, :]

        # function returns the last iteration result as a numpy array and all iteration results in a list
        return absorption, list_of_intermediate_absorptions

    def extract_initial_data_from_hdf5(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract necessary information - initial pressure and scattering coefficients -
        from settings dictionaries. The setting dictionaries is extracted from a hdf5 file.

        :return: Initial pressure and a priori known scattering coefficients.
        """

        # get simulation output which contains initial pressure and scattering
        file = load_hdf5(self.global_settings[Tags.SIMPA_OUTPUT_PATH])
        wavelength = self.global_settings[Tags.WAVELENGTH]

        # get initial pressure and scattering
        image = file["simulations"]["optical_forward_model_output"]["initial_pressure"][str(wavelength)]
        mus = file["simulations"]["simulation_properties"]["mus"][str(wavelength)]

        # function returns the last iteration result as a numpy array and all iteration results in a list
        return image, mus

    def preprocessing_for_iterative_qPAI(self, image_data: np.ndarray,
                                         scattering: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Preprocesses image data and scattering distribution for iterative algorithm using mcx.
        The preprocessing step includes:
        1. Stacking the input data from 2-d to 3-d if necessary, since the mcx adapter can only perform
           a Monte Carlo Simulation of fluence given 3-d volumes of absorption, scattering, and anisotropy
        2. Resampling the input data to mitigate the inverse crime

        :param image_data: Raw input image of initial pressure.
        :param scattering: Map of scattering coefficients known a priori.
        :return: Resampled and (if necessary) stacked volume of noisy
                 initial pressure, scattering and bool indicating if image had to be stacked to 3-d.
        """

        if len(np.shape(image_data)) == 2:
            stacked_to_volume = True
            image_data = self.stacking_to_3d(image_data)
        else:
            stacked_to_volume = False
        if len(np.shape(scattering)) == 2:
            scattering = self.stacking_to_3d(scattering)

        image_data, scattering = self.resampling_for_iterative_qPAI(image_data, scattering)

        return image_data, scattering, stacked_to_volume

    def stacking_to_3d(self, input_data: np.ndarray) -> np.ndarray:
        """
        Stacks the input map in sequence along axis=1 to build 3-d volume.

        :param input_data: 2-d image data.
        :return: Volume of stacked input image.
        """

        spacing = self.global_settings[Tags.DIM_VOLUME_X_MM] / np.shape(input_data)[0]
        num_repeats = int(self.global_settings[Tags.DIM_VOLUME_Y_MM] / spacing)

        stacked_volume = np.stack([input_data] * num_repeats, axis=1)

        return stacked_volume

    def resampling_for_iterative_qPAI(self, image_data: np.ndarray,
                                      scattering: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downscales the input image and scattering map by a given scale to avoid inverse crime.

        :param image_data: Raw input image of initial pressure.
        :param scattering: Map of scattering coefficients.
        :return: Downscaled image and scattering map.
        """

        downscaling_method = "nearest"
        scale = 0.73

        if Tags.DOWNSCALE_FACTOR in self.iterative_method_settings:
            scale = float(self.iterative_method_settings[Tags.DOWNSCALE_FACTOR])

        downscaled_image = zoom(image_data, scale, order=1, mode=downscaling_method)
        downscaled_scattering = zoom(scattering, scale, order=1, mode=downscaling_method)

        new_spacing = self.global_settings[Tags.SPACING_MM] / scale

        if self.global_settings[Tags.SPACING_MM] != new_spacing:
            self.global_settings[Tags.SPACING_MM] = new_spacing

        if Tags.ILLUMINATION_POSITION in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS]:
            pos = [(elem * scale) for elem in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.ILLUMINATION_POSITION]]
            self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.ILLUMINATION_POSITION] = pos

        if Tags.ILLUMINATION_PARAM1 in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS]:
            param1 = [(elem * scale) for elem in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.ILLUMINATION_PARAM1]]
            self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.ILLUMINATION_PARAM1] = param1

        if Tags.ILLUMINATION_PARAM1 in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS]:
            param2 = [(elem * scale) for elem in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.ILLUMINATION_PARAM2]]
            self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.ILLUMINATION_PARAM2] = param2

        return downscaled_image, downscaled_scattering

    def regularization_sigma(self, input_image: np.ndarray) -> [np.ndarray, int, float]:
        """
        Computes spatial (same shape as input image) or constant regularization parameter sigma.

        :param input_image: Noisy input image.
        :return: Regularization parameter.
        :raises: ValueError: if estimated noise is zero, so SNR cannot be computed.
        """
        noise = float(estimate_sigma(input_image))

        if noise == 0.0:
            raise ValueError("An estimated noise level of zero cannot be used to compute a signal to noise ratio.")

        signal_noise_ratio = input_image / noise

        if Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION in self.iterative_method_settings:
            if self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION]:
                sigma = 1e-2
                if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in self.iterative_method_settings:
                    sigma = self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]
            else:
                sigma = 1 / signal_noise_ratio
                sigma[sigma > 1e8] = 1e8
                sigma[sigma < 1e-8] = 1e-8
        else:
            sigma = 1 / signal_noise_ratio
            sigma[sigma > 1e8] = 1e8
            sigma[sigma < 1e-8] = 1e-8

        return sigma

    def standard_optical_properties(self, image_data: np.ndarray) -> dict:
        """
        Returns a optical properties dictionary containing scattering coefficients and anisotropy.

        :param image_data: Measured image data.
        :return: Optical properties (scattering and anisotropy).
        """

        shape = np.shape(image_data)

        # scattering must be known a priori at the moment.
        if Tags.PROPERTY_SCATTERING_PER_CM in self.global_settings:
            mus = float(self.global_settings[Tags.PROPERTY_SCATTERING_PER_CM]) * np.ones(shape)
        else:
            background_dict = TISSUE_LIBRARY.muscle()
            mus = float(MolecularComposition.get_properties_for_wavelength(background_dict, wavelength=800)["mus"])
            mus = mus * np.ones(shape)

        if Tags.PROPERTY_ANISOTROPY in self.global_settings:
            g = float(self.global_settings[Tags.PROPERTY_ANISOTROPY]) * np.ones(shape)
            g[:, :, :] = 0.9  # anisotropy is always 0.9 when simulating
        else:
            g = float(OpticalTissueProperties.STANDARD_ANISOTROPY) * np.ones(shape)

        optical_properties = {
            "scattering": mus,
            "anisotropy": g
        }

        return optical_properties

    def forward_model_fluence(self, absorption: np.ndarray,
                              scattering: np.ndarray, anisotropy: np.ndarray) -> np.ndarray:
        """
        Simulates photon propagation in 3-d volume and returns simulated fluence map in units of J/cm^2.

        :param absorption: Volume of absorption coefficients in 1/cm for Monte Carlo Simulation.
        :param scattering: Volume of scattering coefficients in 1/cm for Monte Carlo Simulation.
        :param anisotropy: Volume of anisotropy data for Monte Carlo Simulation.
        :return: Fluence map.
        :raises: AssertionError: if Tags.OPTICAL_MODEL tag was not or incorrectly defined in settings.
        """

        if Tags.OPTICAL_MODEL not in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS]:
            raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings.")
        model = self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.OPTICAL_MODEL]

        if model == Tags.OPTICAL_MODEL_MCX:
            forward_model_implementation = OpticalForwardModelMcxAdapter(self.global_settings)
        else:
            raise AssertionError("Tags.OPTICAL_MODEL tag must be Tags.OPTICAL_MODEL_MCX.")

        fluence = forward_model_implementation.forward_model(absorption_cm=absorption,
                                                             scattering_cm=scattering,
                                                             anisotropy=anisotropy)

        print("Simulating the optical forward process...[Done]")

        return fluence

    def update_absorption_estimate(self, image_data: np.ndarray, fluence: np.ndarray,
                                   sigma: [np.ndarray, int, float]) -> np.ndarray:
        """
        Reconstructs map of absorption coefficients in 1/cm given measured data and simulated fluence.

        :param image_data: Measured image data (initial pressure) used for reconstruction.
        :param fluence: Simulated fluence map in J/cm^2.
        :param sigma: Regularization factor to avoid instability if the fluence is low.
        :return: Reconstructed absorption.
        """

        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS]:
            if Tags.PROPERTY_GRUNEISEN_PARAMETER in self.global_settings:
                gamma = self.global_settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
            else:
                gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
                gamma = gamma * np.ones(np.shape(image_data))
            factor = (self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
            absorption = np.array(image_data / ((fluence + sigma) * gamma * factor))
        else:
            absorption = image_data / (fluence + sigma)

        return absorption

    def log_sum_squared_error(self, image_data: np.ndarray, absorption: np.ndarray, fluence: np.ndarray,
                              sigma: [np.ndarray, int, float]) -> float:
        """
        Computes log (base 10) of the sum of squared error between image and reconstructed pressure map in middle slice.

        :param image_data: Measured image data used for reconstruction.
        :param absorption: Reconstructed map of absorption coefficients in 1/cm.
        :param fluence: Simulated fluence map in J/cm^2.
        :param sigma: Regularization parameter to avoid instability if the fluence is low.
        :return: sse error.
        """

        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in self.global_settings[Tags.OPTICAL_MODEL_SETTINGS]:
            if Tags.PROPERTY_GRUNEISEN_PARAMETER in self.global_settings:
                gamma = self.global_settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
            else:
                gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
                gamma = gamma * np.ones(np.shape(image_data))
            factor = (self.global_settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
            predicted_pressure = absorption * (fluence + sigma) * gamma * factor
        else:
            predicted_pressure = absorption * (fluence + sigma)

        y_pos = int(image_data.shape[1] / 2)
        sse = np.sum(np.square(image_data[:, y_pos, :] - predicted_pressure[:, y_pos, :]))

        return np.log10(sse)

    def convergence_stopping_criterion(self, errors: list, iteration: int) -> bool:
        """
        Serves as a stopping criterion for the iterative algorithm. If False the iterative algorithm continues.

        :param errors: List of log (base 10) sum of squared errors.
        :param iteration: Iteration number.
        :return: if iteration method should be stopped.
        :raises: AssertionError: if Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL tag is zero
        """
        epsilon = 0.03

        if Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL in self.iterative_method_settings:
            if self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL] == 0:
                raise AssertionError("Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL should be greater than zero.")
            epsilon = self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL]

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
