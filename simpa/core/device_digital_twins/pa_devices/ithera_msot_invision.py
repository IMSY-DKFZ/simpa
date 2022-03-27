# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import numpy as np

from simpa.core.device_digital_twins import PhotoacousticDevice, CurvedArrayDetectionGeometry, \
    MSOTInVisionIlluminationGeometry


class InVision256TF(PhotoacousticDevice):
    """
    This class represents a digital twin of the InVision 256-TF, manufactured by iThera Medical, Munich, Germany
    (https://www.ithera-medical.com/products/msot-invision/). It is based on the real specifications of the device, but
    due to the limitations of the possibilities how to represent a device in the software frameworks,
    constitutes only an approximation.

    Some important publications that showcase the use cases of the InVision series devices are::

        Joseph, James, et al. "Evaluation of precision in optoacoustic tomography
        for preclinical imaging in living subjects."
        Journal of Nuclear Medicine 58.5 (2017): 807-814.

        Merčep, Elena, et al. "Whole-body live mouse imaging by hybrid
        reflection-mode ultrasound and optoacoustic tomography."
        Optics letters 40.20 (2015): 4643-4646.

    """

    def __init__(self, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = np.asarray([-20, 20, 0, 0, -20, 20])):
        """
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of e.g. detector element positions or illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(InVision256TF, self).__init__(device_position_mm=device_position_mm,
                                            field_of_view_extent_mm=field_of_view_extent_mm)

        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.74,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.64,
                                                          detector_element_length_mm=15,
                                                          center_frequency_hz=5e6,
                                                          bandwidth_percent=55,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=0,
                                                          device_position_mm=device_position_mm,
                                                          field_of_view_extent_mm=field_of_view_extent_mm)

        self.field_of_view_extent_mm = detection_geometry.field_of_view_extent_mm
        self.set_detection_geometry(detection_geometry)
        for i in range(10):
            self.add_illumination_geometry(MSOTInVisionIlluminationGeometry(invision_position=self.device_position_mm,
                                                                            geometry_id=i))

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"InVision256TF": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = InVision256TF()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
