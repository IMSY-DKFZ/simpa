# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from simpa.core.device_digital_twins.digital_device_twin_base import DigitalDeviceTwinBase
import numpy as np


class DetectionGeometryBase(DigitalDeviceTwinBase):
    """
    This class is the base class for representing all detector geometries.
    """
    def __init__(self, number_detector_elements, detector_element_width_mm,
                 detector_element_length_mm, center_frequency_hz, bandwidth_percent,
                 sampling_frequency_mhz, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        """

        :param number_detector_elements: Total number of detector elements.
        :type number_detector_elements: int
        :param detector_element_width_mm: In-plane width of one detector element (pitch - distance between two
            elements) in mm.
        :type detector_element_width_mm: int, float
        :param detector_element_length_mm: Out-of-plane length of one detector element in mm.
        :type detector_element_length_mm: int, float
        :param center_frequency_hz: Center frequency of the detector with approximately gaussian frequency response in
            Hz.
        :type center_frequency_hz: int, float
        :param bandwidth_percent: Full width at half maximum in percent of the center frequency.
        :type bandwidth_percent: int, float
        :param sampling_frequency_mhz: Sampling frequency of the detector in MHz.
        :type sampling_frequency_mhz: int, float
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of detector positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(DetectionGeometryBase, self).__init__(device_position_mm=device_position_mm,
                                                    field_of_view_extent_mm=field_of_view_extent_mm)
        self.number_detector_elements = number_detector_elements
        self.detector_element_width_mm = detector_element_width_mm
        self.detector_element_length_mm = detector_element_length_mm
        self.center_frequency_Hz = center_frequency_hz
        self.bandwidth_percent = bandwidth_percent
        self.sampling_frequency_MHz = sampling_frequency_mhz

    @abstractmethod
    def get_detector_element_positions_base_mm(self) -> np.ndarray:
        """
        Defines the abstract positions of the detection elements in an arbitrary coordinate system.
        Typically, the center of the field of view is defined as the origin.

        To obtain the positions in an interpretable coordinate system, please use the other method::

            get_detector_element_positions_accounting_for_device_position_mm

        :returns: A numpy array containing the position vectors of the detection elements.

        """
        pass

    def get_detector_element_positions_accounting_for_device_position_mm(self) -> np.ndarray:
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

    def get_detector_element_positions_accounting_for_field_of_view(self) -> np.ndarray:
        """
        Similar to::

            get_detector_element_positions_base_mm

        This method returns the absolute positions of the detection elements relative to the device
        position in the imaged volume, where the device position is defined by the following tag::

            Tags.DIGITAL_DEVICE_POSITION

        :returns: A numpy array containing the coordinates of the detection elements

        """
        abstract_element_positions = np.copy(self.get_detector_element_positions_base_mm())
        field_of_view = self.field_of_view_extent_mm
        x_half = (field_of_view[1] - field_of_view[0]) / 2
        y_half = (field_of_view[3] - field_of_view[2]) / 2
        if np.abs(x_half) < 1e-10:
            abstract_element_positions[:, 0] = 0
        if np.abs(y_half) < 1e-10:
            abstract_element_positions[:, 1] = 0

        abstract_element_positions[:, 0] += x_half
        abstract_element_positions[:, 1] += y_half
        abstract_element_positions[:, 2] += field_of_view[4]
        return abstract_element_positions

    @abstractmethod
    def get_detector_element_orientations(self) -> np.ndarray:
        """
        This method yields a normalised orientation vector for each detection element. The length of
        this vector is the same as the one obtained via the position methods::

            get_detector_element_positions_base_mm
            get_detector_element_positions_accounting_for_device_position_mm

        :returns: a numpy array that contains normalised orientation vectors for each detection element

        """
        pass

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        return {DetectionGeometryBase: serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = DetectionGeometryBase()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
