# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from simpa.core.device_digital_twins import DigitalDeviceTwinBase
from simpa.utils import Tags
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
    def get_detector_element_surface_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return one width-axis and one length-axis unit vector per detector element
        """
        pass

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

    def get_detector_element_corners_mm(self) -> np.ndarray:
        """
        This method returns four physical corner coordinates of every rectangular detector element.
        """
        centers_mm = np.asarray(self.get_detector_element_positions_accounting_for_device_position_mm(), dtype=float)

        width_axes, length_axes = self.get_detector_element_surface_axes()
        width_axes = np.asarray(width_axes, dtype=float)
        length_axes = np.asarray(length_axes, dtype=float)

        self._validate_detector_surface_axes(width_axes, length_axes)

        half_width_mm = self.detector_element_width_mm / 2
        half_length_mm = self.detector_element_length_mm / 2

        offsets_mm = np.stack(
            [
                -half_width_mm * width_axes - half_length_mm * length_axes,
                -half_width_mm * width_axes + half_length_mm * length_axes,
                half_width_mm * width_axes - half_length_mm * length_axes,
                half_width_mm * width_axes + half_length_mm * length_axes,
            ], axis=1
        )
        return centers_mm[:, np.newaxis, :] + offsets_mm

    def get_physical_detector_bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """
        This method calculates the physical bounds enclosing all detector element surfaces.
        """
        corners_mm = self.get_detector_element_corners_mm()
        lower_mm = corners_mm.min(axis=(0, 1))
        upper_mm = corners_mm.max(axis=(0, 1))
        return lower_mm, upper_mm

    @staticmethod
    def get_acoustic_grid_bounds_mm(global_settings) -> tuple[np.ndarray, np.ndarray]:
        grid_shape = np.asarray(global_settings.get_volume_dimensions_voxels(), dtype=int)
        spacing_mm = float(global_settings[Tags.SPACING_MM])

        lower_mm = np.zeros(3)
        upper_mm = (grid_shape-1) * spacing_mm
        return lower_mm, upper_mm

    def check_detector_elements_inside_acoustic_grid(self, global_settings, tolerance_mm: float=1e-9) -> bool:
        detector_lower_mm, detector_upper_mm = self.get_physical_detector_bounds_mm()
        grid_lower_mm, grid_upper_mm = self.get_acoustic_grid_bounds_mm(global_settings)

        lower_inside = np.all(detector_lower_mm >= grid_lower_mm - tolerance_mm)
        upper_inside = np.all(detector_upper_mm <= grid_upper_mm + tolerance_mm)

        if not lower_inside or not upper_inside:
            try:
                required_dimensions_mm = (self.get_required_volume_dimensions_mm(float(global_settings[Tags.SPACING_MM])))
                recommendation = f"Recommended minimum volume dimensions {required_dimensions_mm} mm."
            except ValueError as error:
                recommendation = str(error)

            self.logger.error(f"Physical detector elements are not fully contained inside the acoustic grid. Detector bounds: {detector_lower_mm}-{detector_upper_mm} mm. Acoustic grid bounds: {grid_lower_mm}-{grid_upper_mm} mm. {recommendation}")
            return False

        return True

    def get_required_volume_dimensions_mm(self, spacing_mm: float, tolerance_mm: float=1e-9) -> np.ndarray:
        """
        This method calculates the required dimensions for all detectors to be contained.
        """
        if spacing_mm <= 0:
            raise ValueError("Spacing must be greater than 0.")

        detector_lower_mm, detector_upper_mm = self.get_physical_detector_bounds_mm()
        if np.any(detector_lower_mm < -tolerance_mm):
            required_translation_mm = np.maximum(-detector_lower_mm, 0.0)
            raise ValueError("Increasing the volume dimensions is not sufficient. Move the device by at least "
                             f"{required_translation_mm} mm before sizing the volume.")
        required_grid_points = (np.ceil((detector_upper_mm - tolerance_mm) / spacing_mm).astype(int) + 1)
        return required_grid_points * spacing_mm

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        return {DetectionGeometryBase: serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = DetectionGeometryBase()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device

    def _validate_detector_surface_axes(self, width_axes: np.ndarray, length_axes: np.ndarray) -> None:
        normals = np.asarray(self.get_detector_element_orientations(), dtype=float)
        expected_shape = (self.number_detector_elements, 3)

        if width_axes.shape != expected_shape:
            raise ValueError(
                f"Detector width axes must have shape {expected_shape}, but have shape {width_axes.shape}."
            )

        if length_axes.shape != expected_shape:
            raise ValueError(
                f"Detector length axes must have shape {expected_shape}, but have shape {length_axes.shape}."
            )

        if not np.all(np.isfinite(width_axes)):
            raise ValueError("Detector width axes contain non-finite values.")

        if not np.all(np.isfinite(length_axes)):
            raise ValueError("Detector length axes contain non-finite values.")

        if not np.allclose(np.linalg.norm(width_axes, axis=1), 1.0, atol=1e-8):
            raise ValueError("Detector width axes must be unit vectors.")

        if not np.allclose(np.linalg.norm(length_axes, axis=1), 1.0, atol=1e-8):
            raise ValueError("Detector length axes must be unit vectors.")

        if not np.allclose(np.sum(width_axes * length_axes, axis=1), 0.0, atol=1e-8):
            raise ValueError("Detector width and length axes must be perpendicular.")

        if not np.allclose(np.sum(width_axes * normals, axis=1), 0.0, atol=1e-8):
            raise ValueError("Detector width axes must be perpendicular to detector normals.")

        if not np.allclose(np.sum(length_axes * normals, axis=1), 0.0, atol=1e-8):
            raise ValueError("Detector length axes must be perpendicular to detector normals.")
