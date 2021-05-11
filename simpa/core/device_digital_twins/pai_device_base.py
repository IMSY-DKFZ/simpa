"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from abc import abstractmethod
from simpa.utils.settings import Settings
import numpy as np
from simpa.log import Logger
from simpa.utils import Tags


class PAIDeviceBase:
    """
    This class represents a PAI device including the detection and illumination geometry.
    """

    def __init__(self):
        self.logger = Logger()
        self.probe_height_mm = 0

    @abstractmethod
    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        """
        It might be that certain device geometries need a certain dimensionality of the simulated PAI volume, or that
        it required the existence of certain Tags in the global global_settings.
        To this end, a  PAI device should use this method to inform the user about a mismatch of the desired device and
        throw a ValueError if that is the case.

        :raises ValueError: raises a value error if the prerequisites are not matched.
        :returns: True if the prerequisites are met, False if they are not met, but no exception has been raised.

        """
        pass

    @abstractmethod
    def adjust_simulation_volume_and_settings(self, global_settings: Settings) -> Settings:
        """
        In case that the PAI device needs space for the arrangement of detectors or illuminators in the volume,
        this method will update the volume accordingly.
        """
        pass

    @abstractmethod
    def get_illuminator_definition(self, global_settings: Settings):
        """
        Defines the illumination geometry of the device in the settings dictionary.
        """
        pass

    @abstractmethod
    def get_detector_element_positions_base_mm(self) -> np.ndarray:
        """
        Defines the abstract positions of the detection elements in an arbitraty coordinate system.
        Typically, the center of the field of view is defined as the origin.

        To obtain the positions in an interpretable coordinate system, please use the other method::

            get_detector_element_positions_accounting_for_device_position_mm

        :returns: A numpy array containing the position vestors of the detection elements.

        """
        pass

    @abstractmethod
    def get_detector_element_positions_accounting_for_device_position_mm(self, global_settings: Settings) -> np.ndarray:
        """
        Similar to::

            get_detector_element_positions_base_mm

        This method returns the absolute positions of the detection elements relative to the device
        position in the imaged volume, where the device position is defined by the following tag::

            Tags.DIGITAL_DEVICE_POSITION

        :returns: A numpy array containing the coordinates of the detection elements

        """
        detector_element_positions_mm = self.get_detector_element_positions_base_mm()

        if Tags.DIGITAL_DEVICE_POSITION in global_settings and global_settings[Tags.DIGITAL_DEVICE_POSITION]:
            device_position = np.asarray(global_settings[Tags.DIGITAL_DEVICE_POSITION])
        else:
            device_position = self.get_default_probe_position(global_settings)

        return np.add(detector_element_positions_mm, device_position)

    @abstractmethod
    def get_detector_element_orientations(self, global_settings: Settings) -> np.ndarray:
        """
        This method yields a normalised orientation vector for each detection element. The length of
        this vector is the same as the one obtained via the position methods::

            get_detector_element_positions_base_mm
            get_detector_element_positions_accounting_for_device_position_mm

        :returns: a numpy array that contains normalised orientation vectors for each detection element

        """
        pass

    @abstractmethod
    def get_default_probe_position(self, global_settings: Settings) -> np.ndarray:
        """
        Returns the default probe position in case none was given in the Settings dict.
        """
        pass
