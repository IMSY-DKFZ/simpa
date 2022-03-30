# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5
from simpa.io_handling.ipasc import export_to_ipasc
from simpa.utils.settings import Settings
from simpa.log import Logger
from .device_digital_twins.digital_device_twin_base import DigitalDeviceTwinBase

import numpy as np
import os
import time


def simulate(simulation_pipeline: list, settings: Settings, digital_device_twin: DigitalDeviceTwinBase):
    """
    This method constitutes the staring point for the simulation pipeline
    of the SIMPA toolkit.

    :param simulation_pipeline: a list of callable functions
    :param settings: settings dictionary containing the simulation instructions
    :param digital_device_twin: a digital device twin of an imaging device as specified by the DigitalDeviceTwinBase
        class.
    :raises TypeError: if one of the given parameters is not of the correct type
    :raises AssertionError: if the digital device twin is not able to simulate the settings specification
    :return: list with the save paths of the simulated data within the HDF5 file.
    """
    start_time = time.time()
    logger = Logger()
    if not isinstance(settings, Settings):
        logger.critical("The second argument was not a settings instance!")
        raise TypeError("Use a Settings instance from simpa.utils.settings_generator as simulation input.")

    if not isinstance(simulation_pipeline, list):
        logger.critical("The first argument was not a list with pipeline methods!")
        raise TypeError("The simulation pipeline must be a list that contains callable functions.")

    if not digital_device_twin.check_settings_prerequisites(settings):
        msg = ("The simulation settings do not work with the digital device twin chosen."
               "Please check the log for details.")
        logger.critical(msg)
        raise AssertionError(msg)

    simpa_output = dict()
    path = settings[Tags.SIMULATION_PATH] + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    if Tags.SIMPA_OUTPUT_NAME in settings:
        simpa_output_path = path + settings[Tags.SIMPA_OUTPUT_NAME]
    else:
        simpa_output_path = path + settings[Tags.VOLUME_NAME]

    settings[Tags.SIMPA_OUTPUT_PATH] = simpa_output_path + ".hdf5"

    simpa_output[Tags.SETTINGS] = settings
    simpa_output[Tags.DIGITAL_DEVICE] = digital_device_twin
    simpa_output[Tags.SIMULATION_PIPELINE] = [type(x).__name__ for x in simulation_pipeline]

    logger.debug("Saving settings dictionary...")
    save_hdf5(simpa_output, settings[Tags.SIMPA_OUTPUT_PATH])
    logger.debug("Saving settings dictionary...[Done]")

    for wavelength in settings[Tags.WAVELENGTHS]:
        logger.debug(f"Running pipeline for wavelength {wavelength}nm...")

        if settings[Tags.RANDOM_SEED] is not None:
            np.random.seed(settings[Tags.RANDOM_SEED])
        else:
            np.random.seed(None)

        settings[Tags.WAVELENGTH] = wavelength

        for pipeline_element in simulation_pipeline:
            logger.debug(f"Running {type(pipeline_element)}")
            pipeline_element.run(digital_device_twin)

        logger.debug(f"Running pipeline for wavelength {wavelength}nm... [Done]")

    # If the dimensions of the simulation results are changed after calling the respective module
    # adapter / processing components, the amount of space on the hard drive that is allocated by the HDF5
    # code does not dynamically change. This can be remedied by re-writing the file after the simulation
    # terminates. As it might have a negative impact on simulation performance, it must be activated
    # by the user manually. Active by default.
    if not(Tags.DO_FILE_COMPRESSION in settings and
            not settings[Tags.DO_FILE_COMPRESSION]):
        all_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH])
        if Tags.VOLUME_CREATION_MODEL_SETTINGS in all_data[Tags.SETTINGS] and \
                Tags.INPUT_SEGMENTATION_VOLUME in all_data[Tags.SETTINGS][Tags.VOLUME_CREATION_MODEL_SETTINGS]:
            del all_data[Tags.SETTINGS][Tags.VOLUME_CREATION_MODEL_SETTINGS][Tags.INPUT_SEGMENTATION_VOLUME]
        save_hdf5(all_data, settings[Tags.SIMPA_OUTPUT_PATH], file_compression="gzip")

    # Export simulation result to the IPASC format.
    if Tags.DO_IPASC_EXPORT in settings and settings[Tags.DO_IPASC_EXPORT]:
        logger.info("Exporting to IPASC....")
        export_to_ipasc(settings[Tags.SIMPA_OUTPUT_PATH], device=digital_device_twin)

    logger.info(f"The entire simulation pipeline required {time.time() - start_time} seconds.")
