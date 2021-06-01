"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from abc import abstractmethod
from simpa.utils import Settings, Tags
from simpa.core.device_digital_twins.digital_device_twin_base import DigitalDeviceTwinBase
import numpy as np


class DetectionGeometryBase(DigitalDeviceTwinBase):
    """
    This class represents an illumination geometry
    """
    def __init__(self, number_detector_elements, detector_element_width_mm,
                 detector_element_length_mm, center_frequency_hz, bandwidth_percent,
                 sampling_frequency_mhz, probe_width_mm, device_position_mm: np.ndarray = None):
        super().__init__(device_position_mm=device_position_mm)
        self.number_detector_elements = number_detector_elements
        self.detector_element_width_mm = detector_element_width_mm
        self.detector_element_length_mm = detector_element_length_mm
        self.center_frequency_Hz = center_frequency_hz
        self.bandwidth_percent = bandwidth_percent
        self.sampling_frequency_MHz = sampling_frequency_mhz
        self.probe_width_mm = probe_width_mm

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

    def get_detector_element_positions_accounting_for_device_position_mm(self, global_settings: Settings) -> np.ndarray:
        """
        Similar to::

            get_detector_element_positions_base_mm

        This method returns the absolute positions of the detection elements relative to the device
        position in the imaged volume, where the device position is defined by the following tag::

            Tags.DIGITAL_DEVICE_POSITION

        :returns: A numpy array containing the coordinates of the detection elements

        """
        abstract_element_positions = self.get_detector_element_positions_base_mm()
        device_position = self.device_position_mm
        return np.add(abstract_element_positions, device_position)

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


