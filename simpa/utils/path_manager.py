# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import inspect
from dotenv import load_dotenv
from pathlib import Path
from simpa.log import Logger


class PathManager:
    """
    As a pipelining tool that serves as a communication layer between different numerical forward models and
    processing tools, SIMPA needs to be configured with the paths to these tools on your local hard drive.
    To this end, we have implemented the `PathManager` class that you can import to your project using
    `from simpa.utils import PathManager`. The PathManager looks for a `path_config.env` file (just like the
    one we provided in the `simpa_examples`) in the following places in this order:

        1. The optional path you give the PathManager
        2. Your $HOME$ directory
        3. The current working directory
        4. The SIMPA home directory path
    """

    def __init__(self, environment_path=None):
        """

        :param environment_path: Per default, the config with the environment variables is located in /HOME/path_config.env
        """
        self.logger = Logger()
        self.path_config_file_name = 'path_config.env'
        if environment_path is None:
            environment_path = os.path.join(str(Path.home()), self.path_config_file_name)
            self.logger.debug(f"Using $HOME$ path to search for config file: {environment_path}")
            if not os.path.exists(environment_path) or not os.path.isfile(environment_path):
                self.logger.debug(f"Did not find path config in $HOME$: {environment_path}")
                environment_path = self.detect_local_path_config()
        else:
            if not os.path.isfile(environment_path):
                self.logger.debug(f"No file was supplied. Assuming a folder was given and looking "
                                  f"for {self.path_config_file_name}")
                environment_path = os.path.join(environment_path, self.path_config_file_name)
            self.logger.debug(f"Using supplied path to search for config file: {environment_path}")

        if environment_path is None or not os.path.exists(environment_path) or not os.path.isfile(environment_path):
            error_message = f"Did not find a { self.path_config_file_name} file in any of the standard directories..."
            self.logger.critical(error_message)
            raise FileNotFoundError(error_message)

        self.environment_path = environment_path
        load_dotenv(environment_path, override=True)

    def detect_local_path_config(self):
        """
        This methods looks in the default local paths for a path_config.env file.
        """

        # Look in current working directory
        self.logger.debug("Searching for path config in current working directory...")
        current_working_directory = os.path.join(os.getcwd(), self.path_config_file_name)
        if os.path.exists(current_working_directory):
            self.logger.debug(f"Found {self.path_config_file_name} in current working directory: "
                              f"{current_working_directory}")
            return current_working_directory

        # Look in the SIMPA base directory
        self.logger.debug("Searching for path config in SIMPA base directory...")
        current_file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        simpa_home = os.path.join(current_file_path, "..", "..", self.path_config_file_name)

        if os.path.exists(simpa_home):
            self.logger.debug(f"Found {self.path_config_file_name} in current working directory: "
                              f"{simpa_home}")
            return simpa_home

        return None

    def get_hdf5_file_save_path(self):
        path = self.get_path_from_environment('SAVE_PATH')
        self.logger.debug(f"Retrieved SAVE_PATH={path}")
        return path

    def get_mcx_binary_path(self):
        path = self.get_path_from_environment('MCX_BINARY_PATH')
        self.logger.debug(f"Retrieved MCX_BINARY_PATH={path}")
        return path

    def get_matlab_binary_path(self):
        path = self.get_path_from_environment('MATLAB_BINARY_PATH')
        self.logger.debug(f"Retrieved MATLAB_BINARY_PATH={path}")
        return path

    def get_path_from_environment(self, env_variable_name):
        env_variable_content = os.environ.get(env_variable_name)
        if env_variable_content is None:
            error_string = f"The desired environment path variable {env_variable_name} is not available in"\
                            f" the given config path {self.environment_path}"
            self.logger.critical(error_string)
            raise FileNotFoundError(error_string)
        return env_variable_content
