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

from abc import abstractmethod
from simpa.utils import Settings, Tags
from simpa.core.device_digital_twins.digital_device_base import DigitalDeviceTwinBase
import numpy as np


class DetectionGeometryBase(DigitalDeviceTwinBase):
    """
    This class represents an illumination geometry
    """
    def __init__(self, pitch_mm, number_detector_elements, detector_element_width_mm,
                 detector_element_length_mm, center_frequency_hz, bandwidth_percent,
                 sampling_frequency_mhz, probe_height_mm):
        super().__init__()
        self.pitch_mm = pitch_mm
        self.number_detector_elements = number_detector_elements
        self.detector_element_width_mm = detector_element_width_mm
        self.detector_element_length_mm = detector_element_length_mm
        self.center_frequency_Hz = center_frequency_hz
        self.bandwidth_percent = bandwidth_percent
        self.sampling_frequency_MHz = sampling_frequency_mhz
        self.probe_height_mm = probe_height_mm
        self.probe_width_mm = self.number_detector_elements * self.pitch_mm

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
        abstract_element_positions = self.get_detector_element_positions_base_mm()

        sizes_mm = np.asarray([global_settings[Tags.DIM_VOLUME_X_MM],
                               global_settings[Tags.DIM_VOLUME_Y_MM],
                               global_settings[Tags.DIM_VOLUME_Z_MM]])

        if Tags.DIGITAL_DEVICE_POSITION in global_settings and global_settings[Tags.DIGITAL_DEVICE_POSITION]:
            device_position = np.asarray(global_settings[Tags.DIGITAL_DEVICE_POSITION])
        else:
            device_position = np.array([sizes_mm[0] / 2, sizes_mm[1] / 2, self.probe_height_mm])

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


class LinearDetector(DetectionGeometryBase):
    """
    This class represents a digital twin of a PA device with a slit as illumination next to a linear detection geometry.

    """

    def __init__(self):
        super().__init__(pitch_mm=0.5,
                         number_detector_elements=100,
                         detector_element_width_mm=0.24,
                         detector_element_length_mm=0.5,
                         center_frequency_hz=3.96e6,
                         bandwidth_percent=55,
                         sampling_frequency_mhz=40,
                         probe_height_mm=0)

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        if global_settings[Tags.VOLUME_CREATOR] != Tags.VOLUME_CREATOR_VERSATILE:
            if global_settings[Tags.DIM_VOLUME_Z_MM] <= (self.probe_height_mm + 1):
                self.logger.error("Volume z dimension is too small to encompass the device in simulation!"
                                  "Must be at least {} mm but was {} mm"
                                  .format((self.probe_height_mm + 1),
                                          global_settings[Tags.DIM_VOLUME_Z_MM]))
                return False
            if global_settings[Tags.DIM_VOLUME_X_MM] <= self.probe_width_mm:
                self.logger.error("Volume x dimension is too small to encompass MSOT device in simulation!"
                                  "Must be at least {} mm but was {} mm"
                                  .format(self.probe_width_mm, global_settings[Tags.DIM_VOLUME_X_MM]))
                return False

        global_settings[Tags.SENSOR_CENTER_FREQUENCY_HZ] = self.center_frequency_Hz
        global_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] = self.sampling_frequency_MHz
        global_settings[Tags.SENSOR_BANDWIDTH_PERCENT] = self.bandwidth_percent

        return True

    def get_detector_element_positions_base_mm(self) -> np.ndarray:

        detector_positions = np.zeros((self.number_detector_elements, 3))

        det_elements = np.arange(-int(self.number_detector_elements / 2),
                                 int(self.number_detector_elements / 2)) * self.pitch_mm

        detector_positions[:, 0] = det_elements

        return detector_positions

    def get_detector_element_orientations(self, global_settings: Settings) -> np.ndarray:
        detector_orientations = np.zeros((self.number_detector_elements, 3))
        detector_orientations[:, 2] = -1
        return detector_orientations

    def get_default_probe_position(self, global_settings: Settings) -> np.ndarray:
        return np.array(0, 0, 0)
