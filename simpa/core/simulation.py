"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

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

    if Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE in settings and \
            settings[Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE]:
        all_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH])
        save_hdf5(all_data, settings[Tags.SIMPA_OUTPUT_PATH], file_compression="gzip")

    # Export simulation result to IPASC-compatible format.
    if Tags.DO_IPASC_EXPORT in settings and settings[Tags.DO_IPASC_EXPORT]:
        logger.info("Exporting to IPASC....")
        export_to_ipasc(settings[Tags.SIMPA_OUTPUT_PATH], device=digital_device_twin)

    logger.info(f"The entire simulation pipeline required {time.time() - start_time} seconds.")
