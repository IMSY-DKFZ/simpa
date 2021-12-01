# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
from simpa.io_handling import load_data_field
from simpa.utils import Tags
from simpa.log import Logger
import numpy as np
from abc import ABC, abstractmethod


class MultispectralProcessingAlgorithm(ABC):
    """
    A MultispectralProcessingAlgorithm class represents an algorithm that works with multispectral input data.
    """

    def __init__(self, global_settings, component_settings_key: str):
        """
        Instantiates a multispectral processing algorithm.

        Per default, this methods loads all data from a certain
        Tags.DATA_FIELD into a data array for all
        Tags.WAVELENGTHS.

        """
        if component_settings_key is None:
            raise KeyError("The component settings must be set for a multispectral"
                           "processing algorithm!")
        self.component_settings = global_settings[component_settings_key]

        if Tags.WAVELENGTHS not in self.component_settings:
            raise KeyError("Tags.WAVELENGTHS must be in the component_settings of a multispectral processing algorithm")

        if Tags.DATA_FIELD not in self.component_settings:
            raise KeyError("Tags.DATA_FIELD must be in the component_settings of a multispectral processing algorithm")

        self.logger = Logger()
        self.global_settings = global_settings
        self.wavelengths = self.component_settings[Tags.WAVELENGTHS]
        self.data_field = self.component_settings[Tags.DATA_FIELD]

        self.data = list()
        for i in range(len(self.wavelengths)):
            self.data.append(load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                                             self.data_field,
                                             self.wavelengths[i]))

        self.data = np.asarray(self.data)
        if Tags.SIGNAL_THRESHOLD in self.component_settings:
            self.data[self.data < self.component_settings[Tags.SIGNAL_THRESHOLD]*np.max(self.data)] = 0

    @abstractmethod
    def run(self):
        """
        This method must be implemented by the multispectral algorithm, such that
        any multispectral algorithm can be executed by invoking the run method.
        """
        pass
