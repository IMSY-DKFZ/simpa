"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
import numpy as np

from simpa.core.device_digital_twins import PhotoacousticDevice, CurvedArrayDetectionGeometry, \
    MSOTInVisionIlluminationGeometry
from simpa.utils.settings import Settings
from simpa.utils import Tags


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

    def __init__(self, device_position_mm: np.ndarray = None):
        super(InVision256TF, self).__init__(device_position_mm=device_position_mm)

        class DetectionGeometry(CurvedArrayDetectionGeometry):

            def __init__(self):
                super(DetectionGeometry, self).__init__(pitch_mm=0.74,
                                                        radius_mm=40,
                                                        number_detector_elements=256,
                                                        detector_element_width_mm=0.64,
                                                        detector_element_length_mm=15,
                                                        center_frequency_hz=5e6,
                                                        bandwidth_percent=55,
                                                        sampling_frequency_mhz=40,
                                                        angular_origin_offset=0,
                                                        device_position_mm=device_position_mm)

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
                field_of_view = self.get_field_of_view_mm()
                x_half = (field_of_view[1] - field_of_view[0]) / 2
                y_half = (field_of_view[3] - field_of_view[2]) / 2
                z_half = (field_of_view[5] - field_of_view[4]) / 2
                if np.abs(x_half) < 1e-10:
                    abstract_element_positions[:, 0] = 0
                if np.abs(y_half) < 1e-10:
                    abstract_element_positions[:, 1] = 0
                if np.abs(z_half) < 1e-10:
                    abstract_element_positions[:, 2] = 0

                abstract_element_positions[:, 0] += x_half
                abstract_element_positions[:, 1] += y_half
                abstract_element_positions[:, 2] += z_half

                return abstract_element_positions

            def get_field_of_view_extent_mm(self) -> np.ndarray:
                return np.asarray([-20, 20,
                                   0, 0,
                                   -20, 20])

        detection_geometry = DetectionGeometry()

        self.set_detection_geometry(detection_geometry)
        for i in range(10):
            self.add_illumination_geometry(MSOTInVisionIlluminationGeometry(i))

    def get_field_of_view_extent_mm(self) -> np.ndarray:
        return np.asarray([-20, 20,
                           0, 0,
                           -20, 20])


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = InVision256TF(device_position_mm=np.asarray([50, 10, 50]))
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 100
    settings[Tags.DIM_VOLUME_Y_MM] = 20
    settings[Tags.DIM_VOLUME_Z_MM] = 100
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))
    positions = device.detection_geometry.get_detector_element_positions_accounting_for_device_position_mm()
    detector_elements = device.detection_geometry.get_detector_element_orientations(global_settings=settings)
    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.show()
