"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
from simpa.io_handling import load_hdf5
from simpa.log import Logger
from abc import ABC, abstractmethod


class MultispectralProcessingAlgorithm(ABC):
    """
    A MultispectralProcessingAlgorithm class represents an algorithm that works with multispectral input data.
    """

    def __init__(self, hdf5_path: (str, dict)):
        """
        Instantiates a multispectral processing algorithm.
        The data that is available to the algorithm will be read from a given hdf5 file.
        In case multiple wavelengths are spread over multiple files, the hdf5_path argument can be a list.

        The data will be available via self.data[i], with i being the ith loaded hdf5 file.

        :param hdf5_path: The path to the hdf5 file
        """
        self.logger = Logger()

        self.data = dict()

        if isinstance(hdf5_path, str):
            try:
                self.data[0] = load_hdf5(hdf5_path)
            except IOError as e:
                self.logger.critical(e)
                raise e
        elif isinstance(hdf5_path, list):
            for idx, hdf5_file_path in enumerate(hdf5_path):
                try:
                    self.data[idx] = load_hdf5(hdf5_file_path)
                except IOError as e:
                    self.logger.critical(e)
                    raise e
        else:
            msg = f"Unsupported type argument {type(hdf5_path)} for hdf5_path. Expected list or str."
            self.logger.critical(msg)
            raise TypeError(msg)


    @abstractmethod
    def run(self):
        """
        This method must be implemented by the multispectral algorithm, such that
        any multispectral algorithm can be executed by invoking the run method.
        """
        pass

