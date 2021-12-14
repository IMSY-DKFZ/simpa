# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.core.device_digital_twins import PhotoacousticDevice, PlanarArrayDetectionGeometry, \
    PencilArrayIlluminationGeometry


class RSOMExplorerP50(PhotoacousticDevice):
    """
    This class represents an approximation of the Raster-scanning Optoacoustic Mesoscopy (RSOM) device
    built by iThera Medical (Munich, Germany). Please refer to the companie's website for more information
    (https://www.ithera-medical.com/products/rsom-explorer-p50/).

    Since simulating thousands of individual forward modeling steps to obtain a single raster-scanned image
    is computationally not feasible, we approximate the process with a device design that has detection elements
    across the entire field of view. Because of this limitation we also need to approximate the light source
    with a homogeneous illumination across the field of view.

    The digital device is modeled based on the reported specifications of the RSOM Explorer P50 system.
    Technical details of the system can be found in the dissertation of Mathias Schwarz
    (https://mediatum.ub.tum.de/doc/1324031/1324031.pdf) and you can find more details on
    use cases of the device in the following literature sources::

        Yew, Yik Weng, et al. "Raster-scanning optoacoustic mesoscopy (RSOM) imaging
        as an objective disease severity tool in atopic dermatitis patients."
        Journal of the American Academy of Dermatology (2020).

        Hindelang, B., et al. "Non-invasive imaging in dermatology and the unique
        potential of raster-scan optoacoustic mesoscopy."
        Journal of the European Academy of Dermatology and Venereology
        33.6 (2019): 1051-1061.

    """

    def __init__(self, element_spacing_mm=0.02,
                 number_elements_x=10,
                 number_elements_y=10,
                 device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        """
        :param element_spacing_mm: Detection element pitch in both x- and y-direction. It is an isotropic spacing \
        assumed.
        :type element_spacing_mm: float
        :param number_elements_x: Number of detection elements in x-direction.
        :type number_elements_x: int
        :param number_elements_y: Number of detection elements in y-direction.
        :type number_elements_y: int
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of e.g. detector element positions or illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(RSOMExplorerP50, self).__init__(device_position_mm=device_position_mm)

        detection_geometry = PlanarArrayDetectionGeometry(pitch_mm=element_spacing_mm,
                                                          number_detector_elements_x=number_elements_x,
                                                          number_detector_elements_y=number_elements_y,
                                                          center_frequency_hz=float(50e6),
                                                          bandwidth_percent=100.0,
                                                          sampling_frequency_mhz=500.0,
                                                          detector_element_width_mm=1,
                                                          detector_element_length_mm=1,
                                                          device_position_mm=device_position_mm,
                                                          field_of_view_extent_mm=field_of_view_extent_mm)

        self.set_detection_geometry(detection_geometry)

        illumination_geometry = PencilArrayIlluminationGeometry(pitch_mm=element_spacing_mm,
                                                                number_illuminators_x=number_elements_x,
                                                                number_illuminators_y=number_elements_y)

        self.add_illumination_geometry(illumination_geometry)

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"RSOMExplorerP50": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = RSOMExplorerP50()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
