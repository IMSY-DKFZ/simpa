# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
import numpy as np
# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ExampleDeviceSlitIlluminationLinearDetector(sp.PhotoacousticDevice):
    """
    This class represents a digital twin of a PA device with a slit as illumination next to a linear detection geometry.

    """

    def __init__(self):
        super().__init__()
        # You can choose your detection geometries from simpa/core/device_digital_twins/detection_geometries
        # You can choose your illumination geometries from simpa/core/device_digital_twins/illumination_geometries

        self.set_detection_geometry(sp.LinearArrayDetectionGeometry())
        self.add_illumination_geometry(sp.SlitIlluminationGeometry())


if __name__ == "__main__":
    device = ExampleDeviceSlitIlluminationLinearDetector()
    settings = sp.Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 20
    settings[Tags.DIM_VOLUME_Y_MM] = 50
    settings[Tags.DIM_VOLUME_Z_MM] = 20
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))

    positions = device.get_detection_geometry().get_detector_element_positions_accounting_for_device_position_mm()
    detector_elements = device.get_detection_geometry().get_detector_element_orientations()
    positions = np.round(positions/settings[Tags.SPACING_MM]).astype(int)
    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.show()
