"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.io_handling.io_hdf5 import save_hdf5
from simpa.io_handling.serialization import SIMPAJSONSerializer
from simpa.utils.settings import Settings
from simpa.log import Logger

import numpy as np
import os
import json
import time


def simulate(simulation_pipeline: list, settings: Settings):
    """
    This method constitutes the staring point for the simulation pipeline
    of the SIMPA toolkit. It calls all relevant and wanted simulation modules in the
    following pre-determined order::

        def simulation(settings):
            for wavelength in settings[Tags.WAVELENGTHS]:

                simulation_data = volume_creator.create_simulation_volumes(settings)
                if optical_simulation_module in settings:
                    optical_model.simulate(simulation_data, settings)
                if acoustic_forward_module in settings:
                    acoustic_model.simulate(simulation_data, settings)
                if noise_simulation in settings:
                    noise_model.simulate(simulation_data, settings)
                if reconstruction_module in settings:
                    reconstruction_model.simulate(simulation_data, settings)

                io_handler.save_hdf5(simulation_data, settings)

    :param simulation_pipeline: a list of callable functions
    :param settings: settings dictionary containing the simulation instructions
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

    simpa_output = dict()
    path = settings[Tags.SIMULATION_PATH] + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    if Tags.SIMPA_OUTPUT_NAME in settings:
        simpa_output_path = path + settings[Tags.SIMPA_OUTPUT_NAME]
    else:
        simpa_output_path = path + settings[Tags.VOLUME_NAME]

    serializer = SIMPAJSONSerializer()

    if Tags.SETTINGS_JSON in settings:
        if settings[Tags.SETTINGS_JSON]:
            with open(simpa_output_path + ".json", "w") as json_file:
                json.dump(settings, json_file, indent="\t", default=serializer.default)
            settings[Tags.SETTINGS_JSON_PATH] = simpa_output_path + ".json"

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
            pipeline_element.run()

        logger.debug(f"Running pipeline for wavelength {wavelength}nm... [Done]")

    logger.info(f"The entire simulation pipeline required {time.time() - start_time} seconds.")
