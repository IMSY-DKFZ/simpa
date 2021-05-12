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

    def __init__(self):
        super(InVision256TF, self).__init__()

        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.74,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.64,
                                                          detector_element_length_mm=15,
                                                          center_frequency_hz=5e6,
                                                          bandwidth_percent=55,
                                                          sampling_frequency_mhz=40,
                                                          focus_in_field_of_view_mm=np.array([0, 0, 4]),
                                                          angular_origin_offset=0)

        self.set_detection_geometry(detection_geometry)
        for i in range(10):
            self.add_illumination_geometry(MSOTInVisionIlluminationGeometry(i))

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        return True

    def get_default_probe_position(self, global_settings: Settings) -> np.ndarray:
        sizes_mm = np.asarray([global_settings[Tags.DIM_VOLUME_X_MM],
                               global_settings[Tags.DIM_VOLUME_Y_MM],
                               global_settings[Tags.DIM_VOLUME_Z_MM]])
        return np.array([sizes_mm[0] / 2, sizes_mm[1] / 2, sizes_mm[2] / 2])


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = InVision256TF()
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 100
    settings[Tags.DIM_VOLUME_Y_MM] = 20
    settings[Tags.DIM_VOLUME_Z_MM] = 100
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))
    positions = device.detection_geometry.get_detector_element_positions_accounting_for_device_position_mm(settings)
    detector_elements = device.detection_geometry.get_detector_element_orientations(global_settings=settings)
    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.show()
