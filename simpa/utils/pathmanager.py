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

import os
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
        if environment_path is None:
            environment_path = str(Path.home()) + '/path_config.env'
        self.environment_path = environment_path
        load_dotenv(environment_path)
        self.logger = Logger()

    def get_hdf5_file_save_path(self):
        return self.get_path_from_environment('SAVE_PATH')

    def get_mcx_binary_path(self):
        return self.get_path_from_environment('MCX_BINARY_PATH')

    def get_matlab_binary_path(self):
        return self.get_path_from_environment('MATLAB_BINARY_PATH')

    def get_acoustic_script_path(self):
        return self.get_path_from_environment('ACOUSTIC_MODEL_SCRIPT')

    def get_path_from_environment(self, env_variable_name):
        env_variable_content = os.environ.get(env_variable_name)
        if env_variable_content is None:
            error_string = f"The desired environment path variable {env_variable_name} is not available in"\
                            f" the given config path {self.environment_path}"
            self.logger.critical(error_string)
            raise FileNotFoundError(error_string)
        return env_variable_content
