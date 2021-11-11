# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.device_digital_twins import PhotoacousticDevice
from simpa.utils import Tags, Settings
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def visualise_device(device: PhotoacousticDevice, save_path=None):
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 100
    settings[Tags.DIM_VOLUME_Y_MM] = 20
    settings[Tags.DIM_VOLUME_Z_MM] = 100
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}

    positions = device.detection_geometry.get_detector_element_positions_accounting_for_device_position_mm()
    detector_elements = device.detection_geometry.get_detector_element_orientations()

    plt.figure(figsize=(8, 4))
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
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
