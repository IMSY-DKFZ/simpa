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

    def __init__(self, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = np.asarray([-20, 20, 0, 0, -20, 20])):
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
            self.add_illumination_geometry(MSOTInVisionIlluminationGeometry(i))


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
    detector_elements = device.detection_geometry.get_detector_element_orientations()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("In volume")
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    fov = device.detection_geometry.get_field_of_view_mm()
    plt.plot([fov[0], fov[1], fov[1], fov[0], fov[0]], [fov[4], fov[4], fov[5], fov[5], fov[4]], color="red")
    plt.subplot(1, 2, 2)
    plt.title("Baseline")
    positions = device.detection_geometry.get_detector_element_positions_base_mm()
    fov = device.detection_geometry.field_of_view_extent_mm
    plt.plot([fov[0], fov[1], fov[1], fov[0], fov[0]], [fov[4], fov[4], fov[5], fov[5], fov[4]], color="red")
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.show()
