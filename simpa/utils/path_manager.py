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

import os, inspect
from dotenv import load_dotenv
from pathlib import Path
from simpa.log import Logger


class PathManager:
    """
    The path manager is in charge of setting the simulation paths, such as the executables of the forward models.

    Per default, the config with the environment variables is located in /HOME/path_config.env
    """
    def __init__(self, environment_path=None):
        """

        :param environment_path: Per default, the config with the environment variables is located in /HOME/path_config.env
        """
        self.logger = Logger()
        self.path_config_file_name = '/path_config.env'
        if environment_path is None:
            environment_path = str(Path.home()) + self.path_config_file_name
            self.logger.debug(f"Using $HOME$ path to search for config file: {environment_path}")
            if not os.path.exists(environment_path) or not os.path.isfile(environment_path):
                environment_path = self.detect_local_path_config()
        else:
            if not os.path.isfile(environment_path):
                self.logger.debug(f"No file was supplied. Assuming a folder was given and looking "
                                  f"for {self.path_config_file_name}")
                environment_path = environment_path + "/" + self.path_config_file_name
            self.logger.debug(f"Using supplied path to search for config file: {environment_path}")

        if environment_path is None or not os.path.exists(environment_path) or not os.path.isfile(environment_path):
            error_message = f"Did not find a { self.path_config_file_name} file in any of the standard directories..."
            self.logger.critical(error_message)
            raise FileNotFoundError(error_message)

        self.environment_path = environment_path
        load_dotenv(environment_path)

    def detect_local_path_config(self):
        """
        This methods looks in the default local paths for a path_config.env file.
        """

        # Look in current working directory
        self.logger.debug("Searching for path config in current working directory...")
        current_working_directory = os.getcwd() + "/" + self.path_config_file_name
        if os.path.exists(current_working_directory):
            self.logger.debug(f"Found {self.path_config_file_name} in current working directory: "
                              f"{current_working_directory}")
            return current_working_directory

        # Look in the SIMPA base directory
        self.logger.debug("Searching for path config in SIMPA base directory...")
        current_file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        simpa_home = current_file_path + "/../../" + self.path_config_file_name

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
