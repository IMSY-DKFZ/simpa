# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
from simpa.log import Logger
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import compute_image_dimensions
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
    xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = compute_image_dimensions(device.get_detection_geometry().field_of_view_extent_mm, settings[Tags.SPACING_MM], Logger())

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.gca().add_patch(Rectangle((xdim_start ,ydim_start), xdim, -ydim, linewidth=1, edgecolor='r', facecolor='r', alpha=.5))
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.show()
