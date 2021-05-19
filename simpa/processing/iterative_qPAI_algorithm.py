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
from simpa.io_handling import save_data_field, load_data_field
from simpa.utils import TISSUE_LIBRARY
from simpa.core.simulation_components import ProcessingComponent


class IterativeqPAIProcessingComponent(ProcessingComponent):
    """
        Applies iterative qPAI Algorithm [1] on simulated initial pressure map and saves the
        reconstruction result in the hdf5 output file. If a 2-d map of initial_pressure is passed the algorithm saves
        the reconstructed absorption coefficients as a 2-d map, else a 3-d absorption reconstruction is
        saved.
        The reconstruction result is saved as an image processing entry "iterative_qpai_result" in the hdf5 output file.
        If intended (e.g. for testing) a list of intermediate iteration updates (only 2-d middle slices) can be saved
        as a npy file.
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

        # must be extracted due to resampling
        self.original_spacing = global_settings[Tags.SPACING_MM]
        if Tags.DOWNSCALE_FACTOR in self.iterative_method_settings:
            self.downscale_factor = self.iterative_method_settings[Tags.DOWNSCALE_FACTOR]
        else:
            self.downscale_factor = 0.73

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
        self.logger.debug(f"Resampling factor: {self.downscale_factor}")

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

        # bypass JSON-dump error by manually caching seeds as int()
        if Tags.RANDOM_SEED in self.global_settings:
            self.global_settings[Tags.RANDOM_SEED] = int(self.global_settings[Tags.RANDOM_SEED])
        if Tags.MCX_SEED in self.optical_settings:
            self.global_settings[Tags.MCX_SEED] = int(self.global_settings[Tags.MCX_SEED])

        # run reconstruction
        reconstructed_absorption, list_of_intermediate_absorptions = self.iterative_absorption_reconstruction()

        # make sure that settings are not changed due to resampling
        if self.global_settings[Tags.SPACING_MM] != self.original_spacing:
            self.global_settings[Tags.SPACING_MM] = self.original_spacing

        if Tags.ILLUMINATION_POSITION in self.optical_settings:
            pos = [(elem / self.downscale_factor) for elem in
                   self.optical_settings[Tags.ILLUMINATION_POSITION]]
            self.optical_settings[Tags.ILLUMINATION_POSITION] = pos

        if Tags.ILLUMINATION_PARAM1 in self.optical_settings:
            param1 = [(elem / self.downscale_factor) for elem in
                      self.optical_settings[Tags.ILLUMINATION_PARAM1]]
            self.optical_settings[Tags.ILLUMINATION_PARAM1] = param1

        if Tags.ILLUMINATION_PARAM2 in self.optical_settings:
            param2 = [(elem / self.downscale_factor) for elem in
                      self.optical_settings[Tags.ILLUMINATION_PARAM2]]
            self.optical_settings[Tags.ILLUMINATION_PARAM2] = param2

        # save absorption update of last iteration step in hdf5 data field
        # bypass wavelength error resulting from loading settings from existing hdf5 file
        if Tags.WAVELENGTH in self.global_settings:
            wavelength = self.global_settings[Tags.WAVELENGTH]
        else:
            wavelength = self.global_settings[Tags.WAVELENGTHS][0]
        data_field = Tags.ITERATIVE_qPAI_RESULT
        save_data_field(reconstructed_absorption, self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                        data_field, wavelength)

        # save a list of all intermediate absorption (2-d only) updates in npy file if intended
        # (e.g. for testing algorithm)
        if Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS in self.iterative_method_settings:
            if self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS]:
                dst = self.global_settings[Tags.SIMULATION_PATH] + "/List_reconstructed_qpai_absorptions_" \
                      + str(wavelength) + "_"
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
        image_data, scattering, stacked_to_volume = self.preprocessing_for_iterative_qpai(image_data=image_data,
                                                                                          scattering=scattering)

        # get optical properties necessary for simulation
        optical_properties_dict = self.standard_optical_properties(image_data)
        anisotropy = optical_properties_dict["anisotropy"]

        # regularization parameter sigma
        sigma = self.regularization_sigma(image_data, stacked_to_volume)

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
        # bypass wavelength error resulting from loading settings from existing hdf5 file
        if Tags.WAVELENGTH in self.global_settings:
            wavelength = self.global_settings[Tags.WAVELENGTH]
        else:
            wavelength = self.global_settings[Tags.WAVELENGTHS][0]
        self.logger.debug(f"Wavelength: {wavelength}")
        # get initial pressure and scattering
        image = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
                                wavelength)
        mus = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.PROPERTY_SCATTERING_PER_CM,
                              wavelength)

        # function returns the last iteration result as a numpy array and all iteration results in a list
        return image, mus

    def preprocessing_for_iterative_qpai(self, image_data: np.ndarray,
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
            light_source = self.optical_settings[Tags.ILLUMINATION_TYPE]
            if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in self.iterative_method_settings:
                sigma = self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]
            else:
                sigma = 1e-2
            self.logger.critical("Input data is 2 dimensional and will be stacked to 3 dimensions. "
                                 "Algorithm is attempted with a %s illumination source and a "
                                 "constant sigma of %s. User caution is advised!" %(light_source, sigma))
            stacked_to_volume = True
            image_data = self.stacking_to_3d(image_data)
            scattering = self.stacking_to_3d(scattering)
        else:
            stacked_to_volume = False

        image_data, scattering = self.resampling_for_iterative_qpai(image_data, scattering)

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

    def resampling_for_iterative_qpai(self, image_data: np.ndarray,
                                      scattering: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downscales the input image and scattering map by a given scale (downscale factor) to avoid inverse crime.

        :param image_data: Raw input image of initial pressure.
        :param scattering: Map of scattering coefficients.
        :return: Downscaled image and scattering map.
        """

        downscaling_method = "nearest"

        downscaled_image = zoom(image_data, self.downscale_factor, order=1, mode=downscaling_method)
        downscaled_scattering = zoom(scattering, self.downscale_factor, order=1, mode=downscaling_method)

        new_spacing = self.global_settings[Tags.SPACING_MM] / self.downscale_factor

        if self.global_settings[Tags.SPACING_MM] != new_spacing:
            self.global_settings[Tags.SPACING_MM] = new_spacing

        if Tags.ILLUMINATION_POSITION in self.optical_settings:
            pos = [(elem * self.downscale_factor) for elem in
                   self.optical_settings[Tags.ILLUMINATION_POSITION]]
            self.optical_settings[Tags.ILLUMINATION_POSITION] = pos

        if Tags.ILLUMINATION_PARAM1 in self.optical_settings:
            param1 = [(elem * self.downscale_factor) for elem in
                      self.optical_settings[Tags.ILLUMINATION_PARAM1]]
            self.optical_settings[Tags.ILLUMINATION_PARAM1] = param1

        if Tags.ILLUMINATION_PARAM2 in self.optical_settings:
            param2 = [(elem * self.downscale_factor) for elem in
                      self.optical_settings[Tags.ILLUMINATION_PARAM2]]
            self.optical_settings[Tags.ILLUMINATION_PARAM2] = param2

        return downscaled_image, downscaled_scattering

    def regularization_sigma(self, input_image: np.ndarray, stacked_to_volume) -> [np.ndarray, int, float]:
        """
        Computes spatial (same shape as input image) or constant regularization parameter sigma.

        :param input_image: Noisy input image.
        :param stacked_to_volume: If True input data was 2 dimensional and a constant sigma should be used.
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
                self.logger.debug(f"Regularization parameter: {sigma}")
            elif stacked_to_volume:
                sigma = 1e-2
                if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in self.iterative_method_settings:
                    sigma = self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]
                self.logger.debug(f"Regularization parameter: {sigma}")
            else:
                self.logger.debug("Regularization: SNR/spatially dependent")
                sigma = 1 / signal_noise_ratio
                sigma[sigma > 1e8] = 1e8
                sigma[sigma < 1e-8] = 1e-8
        elif stacked_to_volume:
            sigma = 1e-2
            if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in self.iterative_method_settings:
                sigma = self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]
            self.logger.debug(f"Regularization parameter: {sigma}")
        else:
            self.logger.debug("Regularization: SNR/spatially dependent")
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

        if Tags.OPTICAL_MODEL not in self.optical_settings:
            raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings.")
        model = self.optical_settings[Tags.OPTICAL_MODEL]

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

        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in self.optical_settings:
            if Tags.PROPERTY_GRUNEISEN_PARAMETER in self.global_settings:
                gamma = self.global_settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
            else:
                gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
                gamma = gamma * np.ones(np.shape(image_data))
            factor = (self.optical_settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
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

        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in self.optical_settings:
            if Tags.PROPERTY_GRUNEISEN_PARAMETER in self.global_settings:
                gamma = self.global_settings[Tags.PROPERTY_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
            else:
                gamma = calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS)
                gamma = gamma * np.ones(np.shape(image_data))
            factor = (self.optical_settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000) * 1e6
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
