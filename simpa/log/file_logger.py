# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path
import sys
from simpa.utils.serializer import SerializableSIMPAClass


class Logger(SerializableSIMPAClass):
    """
    The SIMPA Logger.
    The purpose of this class is to guarantee that the logging Config has been set and that logging strings are written
    to the same file throughout the entire simulation pipeline.
    Per default, the log file is located in the home directory as defined by Path.home().

    The log levels are defined the same way they are in the python logging module:
    DEBUG: Detailed information, typically of interest only when diagnosing problems.
    INFO: Confirmation that things are working as expected.
    WARNING: An indication that something unexpected happened, or indicative of some problem in the near future
    (e.g. ‘disk space low’). The software is still working as expected.
    ERROR: Due to a more serious problem, the software has not been able to perform some function.
    CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
    """
    _instance = None
    _simpa_logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _simpa_default_logging_path = str(Path.home())+"/simpa.log"
    _logger = None

    def __new__(cls, path=None, force_new_instance=False, startup_verbose=False):
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

            if startup_verbose:
                cls._logger.debug("##############################")
                cls._logger.debug("NEW SIMULATION SESSION STARTED")
                cls._logger.debug("##############################")

        return cls._instance

    def debug(self, msg):
        """
        Logs a debug message to the logging system.
        DEBUG: Detailed information, typically of interest only when diagnosing problems.

        :param msg: the message to log
        """
        self._logger.debug(msg)

    def info(self, msg):
        """
        Logs an info message to the logging system.
        INFO: Confirmation that things are working as expected.

        :param msg: the message to log
        """
        self._logger.info(msg)

    def warning(self, msg):
        """
        Logs a warning message to the logging system.
        WARNING: An indication that something unexpected happened, or indicative of some problem
        in the near future (e.g. ‘disk space low’). The software is still working as expected.

        :param msg: the message to log
        """
        self._logger.warning(msg)

    def error(self, msg):
        """
        Logs an error message to the logging system.
        ERROR: Due to a more serious problem, the software has not been able to perform some function.

        :param msg: the message to log
        """
        self._logger.error(msg)

    def critical(self, msg):
        """
        Logs a critical message to the logging system.
        CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

        :param msg: the message to log
        """
        self._logger.critical(msg)

    def serialize(self) -> dict:
        return {"Logger": {"Logger": 1}}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_logger = Logger()
        return deserialized_logger
