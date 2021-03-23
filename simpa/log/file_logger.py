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

import logging
from pathlib import Path
import sys


class Logger:
    """
    The SIMPA Logger.
    The purpose of this class is to guarantee that the logging Config has been set and that logging strings are written
    to the same file throughout the entire simulation pipeline.
    Per default, the log file is
    """
    _instance = None
    _simpa_logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _simpa_default_logging_path = str(Path.home())+"/simpa.log"
    _logger = None

    def __new__(cls, path=None, force_new_instance=False):
        # This pattern can be used to realise a singleton implementation in Python
        if cls._instance is None or force_new_instance:
            cls._instance = super(Logger, cls).__new__(cls)

            if path is None:
                path = cls._simpa_default_logging_path

            cls._logger = logging.getLogger("SIMPA Logger")
            cls._logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler(stream=sys.stdout)
            file_handler = logging.FileHandler(path, mode="w")

            console_handler.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)

            console_handler.setFormatter(cls._simpa_logging_formatter)
            file_handler.setFormatter(cls._simpa_logging_formatter)

            cls._logger.addHandler(console_handler)
            cls._logger.addHandler(file_handler)

            cls._logger.debug("##############################")
            cls._logger.debug("NEW SIMULATION SESSION STARTED")
            cls._logger.debug("##############################")

        return cls._instance

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)
