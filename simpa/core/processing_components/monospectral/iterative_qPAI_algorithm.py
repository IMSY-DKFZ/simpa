# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.ndimage import zoom
from skimage.restoration import estimate_sigma
import time
from typing import Tuple
from simpa.utils import Tags
from simpa.utils.libraries.literature_values import OpticalTissueProperties, StandardProperties
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
from simpa.core.simulation_modules.optical_simulation_module.optical_forward_model_mcx_adapter import \
    MCXAdapter
from simpa.utils import Settings
from simpa.io_handling import save_data_field, load_data_field
from simpa.utils import TISSUE_LIBRARY
from simpa.core.processing_components import ProcessingComponent
import os


class IterativeqPAI(ProcessingComponent):
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

    def run(self, pa_device):
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
            self.optical_settings[Tags.MCX_SEED] = int(self.optical_settings[Tags.MCX_SEED])

        # run reconstruction
        reconstructed_absorption, list_of_intermediate_absorptions = self.iterative_absorption_reconstruction(pa_device)

        # make sure that settings are not changed due to resampling
        if self.global_settings[Tags.SPACING_MM] != self.original_spacing:
            self.global_settings[Tags.SPACING_MM] = self.original_spacing

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

    def iterative_absorption_reconstruction(self, pa_device) -> Tuple[np.ndarray, list]:
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
        target_intial_pressure, scattering, anisotropy = self.extract_initial_data_from_hdf5()

        # checking input data
        if not isinstance(target_intial_pressure, np.ndarray):
            raise TypeError("Image data is not a numpy ndarray.")
        elif target_intial_pressure.size == 0:
            raise ValueError("Image data is empty.")
        elif (len(target_intial_pressure.shape) < 2) or (len(target_intial_pressure.shape) > 3):
            raise ValueError("Image data is invalid. Data must be two or three dimensional.")

        if not isinstance(scattering, np.ndarray):
            raise TypeError("Scattering input is not a numpy ndarray.")
        elif scattering.shape != target_intial_pressure.shape:
            raise ValueError("Shape of scattering data is invalid. Scattering must have the same shape as image_data.")

        # get optical properties necessary for simulation
        optical_properties_dict = self.standard_optical_properties(target_intial_pressure)
        if scattering is None:
            scattering = optical_properties_dict["scattering"]
        if anisotropy is None:
            anisotropy = optical_properties_dict["anisotropy"]

        # preprocessing for iterative qPAI method and mcx_adapter
        target_intial_pressure, scattering, anisotropy, stacked_to_volume = self.preprocessing_for_iterative_qpai(
            intial_pressure=target_intial_pressure,
            scattering=scattering,
            anisotropy=anisotropy)

        # regularization parameter sigma
        sigma = self.regularization_sigma(target_intial_pressure, stacked_to_volume)

        # initialization
        absorption = 1e-16 * np.ones(np.shape(target_intial_pressure))
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
            fluence = self.forward_model_fluence(absorption, scattering, anisotropy, pa_device)
            error_list.append(self.log_sum_squared_error(target_intial_pressure, absorption, fluence, sigma))
            absorption = self.update_absorption_estimate(target_intial_pressure, fluence, sigma)

            est_p0 = absorption * fluence

            # only store middle slice (2-d image instead of 3-d volume) in iteration list for better performance
            list_of_intermediate_absorptions.append(absorption[:, y_pos, :])

            # check if current error did not change significantly in comparison to preceding error
            if self.convergence_stopping_criterion(error_list, iteration=i):
                if Tags.ITERATIVE_RECONSTRUCTION_SAVE_LAST_FLUENCE:
                    dst = self.global_settings[Tags.SIMULATION_PATH] + "/last_fluence" + "_"
                    np.save(dst + self.global_settings[Tags.VOLUME_NAME] + ".npy", fluence)
                break
            i += 1

        print("--- %s seconds/iteration ---" % round((time.time() - start_time) / (i + 1), 2))

        # extracting field of view if input initial pressure was passed as a 2-d array
        if stacked_to_volume:
            absorption = absorption[:, y_pos, :]

        # function returns the last iteration result as a numpy array and all iteration results in a list
        return absorption, list_of_intermediate_absorptions

    def extract_initial_data_from_hdf5(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        initial_pressure = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                                           Tags.DATA_FIELD_INITIAL_PRESSURE,
                                           wavelength)
        scattering = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_SCATTERING_PER_CM,
                                     wavelength)

        anisotropy = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_ANISOTROPY,
                                     wavelength)

        # function returns the last iteration result as a numpy array and all iteration results in a list
        return initial_pressure, scattering, anisotropy

    def preprocessing_for_iterative_qpai(self, intial_pressure: np.ndarray,
                                         scattering: np.ndarray,
                                         anisotropy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Preprocesses image data and scattering distribution for iterative algorithm using mcx.
        The preprocessing step includes:
        1. Stacking the input data from 2-d to 3-d if necessary, since the mcx adapter can only perform
           a Monte Carlo Simulation of fluence given 3-d volumes of absorption, scattering, and anisotropy
        2. Resampling the input data to mitigate the inverse crime

        :param intial_pressure: Raw input image of initial pressure.
        :param scattering: Map of scattering coefficients known a priori.
        :return: Resampled and (if necessary) stacked volume of noisy
                 initial pressure, scattering and bool indicating if image had to be stacked to 3-d.
        """

        if len(np.shape(intial_pressure)) == 2:
            if Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA in self.iterative_method_settings:
                sigma = self.iterative_method_settings[Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA]
            else:
                sigma = 1e-2
            self.logger.warning("Input data is 2 dimensional and will be stacked to 3 dimensions. "
                                "Algorithm is attempted with a %s illumination source and a "
                                f"constant sigma of {sigma}. User caution is advised!")
            stacked_to_volume = True
            intial_pressure = self.stacking_to_3d(intial_pressure)
            scattering = self.stacking_to_3d(scattering)
        else:
            stacked_to_volume = False

        intial_pressure, scattering, anisotropy = self.resampling_for_iterative_qpai(intial_pressure, scattering,
                                                                                     anisotropy)

        return intial_pressure, scattering, anisotropy, stacked_to_volume

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

    def resampling_for_iterative_qpai(self, initial_pressure: np.ndarray,
                                      scattering: np.ndarray,
                                      anisotropy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Downscales the input image and scattering map by a given scale (downscale factor) to avoid inverse crime.

        :param initial_pressure: Raw input image of initial pressure.
        :param scattering: Map of scattering coefficients.
        :param anisotropy: Map of anisotropies.
        :return: Downscaled image and scattering map.
        """

        downscaling_method = "nearest"

        downscaled_initial_pressure = zoom(initial_pressure, self.downscale_factor, order=0, mode=downscaling_method)
        downscaled_scattering = zoom(scattering, self.downscale_factor, order=0, mode=downscaling_method)
        downscaled_anisotropy = zoom(anisotropy, self.downscale_factor, order=0, mode=downscaling_method)

        new_spacing = self.global_settings[Tags.SPACING_MM] / self.downscale_factor
        self.global_settings[Tags.SPACING_MM] = new_spacing

        return downscaled_initial_pressure, downscaled_scattering, downscaled_anisotropy

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
        if Tags.DATA_FIELD_SCATTERING_PER_CM in self.global_settings:
            scattering = float(self.global_settings[Tags.DATA_FIELD_SCATTERING_PER_CM]) * np.ones(shape)
        else:
            background_dict = TISSUE_LIBRARY.muscle()
            scattering = float(MolecularComposition.get_properties_for_wavelength(background_dict,
                                                                                  wavelength=800)["mus"])
            scattering = scattering * np.ones(shape)

        if Tags.DATA_FIELD_ANISOTROPY in self.global_settings:
            anisotropy = float(self.global_settings[Tags.DATA_FIELD_ANISOTROPY]) * np.ones(shape)
        else:
            anisotropy = float(OpticalTissueProperties.STANDARD_ANISOTROPY) * np.ones(shape)

        optical_properties = {
            "scattering": scattering,
            "anisotropy": anisotropy
        }

        return optical_properties

    def forward_model_fluence(self, absorption: np.ndarray,
                              scattering: np.ndarray, anisotropy: np.ndarray,
                              pa_device) -> np.ndarray:
        """
        Simulates photon propagation in 3-d volume and returns simulated fluence map in units of J/cm^2.

        :param absorption: Volume of absorption coefficients in 1/cm for Monte Carlo Simulation.
        :param scattering: Volume of scattering coefficients in 1/cm for Monte Carlo Simulation.
        :param anisotropy: Volume of anisotropy data for Monte Carlo Simulation.
        :param pa_device: The simulation device.
        :return: Fluence map.
        :raises: AssertionError: if Tags.OPTICAL_MODEL tag was not or incorrectly defined in settings.
        """

        if Tags.OPTICAL_MODEL not in self.optical_settings:
            raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings.")
        model = self.optical_settings[Tags.OPTICAL_MODEL]

        self.global_settings.get_optical_settings()[Tags.MCX_ASSUMED_ANISOTROPY] = np.mean(anisotropy)

        if model == Tags.OPTICAL_MODEL_MCX:
            forward_model_implementation = MCXAdapter(self.global_settings)
        else:
            raise AssertionError("Tags.OPTICAL_MODEL tag must be Tags.OPTICAL_MODEL_MCX.")

        _device = pa_device.get_illumination_geometry()

        if isinstance(_device, list):
            # per convention this list has at least two elements
            results = forward_model_implementation.forward_model(absorption_cm=absorption,
                                                                 scattering_cm=scattering,
                                                                 anisotropy=anisotropy,
                                                                 illumination_geometry=_device[0])
            fluence = results[Tags.DATA_FIELD_FLUENCE]
            for idx in range(1, len(_device)):
                # we already looked at the 0th element, so go from 1 to n-1
                results = forward_model_implementation.forward_model(absorption_cm=absorption,
                                                                     scattering_cm=scattering,
                                                                     anisotropy=anisotropy,
                                                                     illumination_geometry=_device[idx + 1])
                fluence += results[Tags.DATA_FIELD_FLUENCE]

            fluence = fluence / len(_device)

        else:
            results = forward_model_implementation.forward_model(absorption_cm=absorption,
                                                                 scattering_cm=scattering,
                                                                 anisotropy=anisotropy,
                                                                 illumination_geometry=_device)
            fluence = results[Tags.DATA_FIELD_FLUENCE]

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
            if Tags.DATA_FIELD_GRUNEISEN_PARAMETER in self.global_settings:
                gamma = self.global_settings[Tags.DATA_FIELD_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
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
            if Tags.DATA_FIELD_GRUNEISEN_PARAMETER in self.global_settings:
                gamma = self.global_settings[Tags.DATA_FIELD_GRUNEISEN_PARAMETER] * np.ones(np.shape(image_data))
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
