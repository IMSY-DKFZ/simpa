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

from simpa.core.device_digital_twins import PhotoacousticDevice, PlanarArrayDetectionGeometry, \
    PencilArrayIlluminationGeometry
from simpa.utils.settings import Settings
from simpa.utils import Tags


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
                 number_elements_y=10):
        super(RSOMExplorerP50, self).__init__()

        detection_geometry = PlanarArrayDetectionGeometry(pitch_mm=element_spacing_mm,
                                                          number_detector_elements_x=number_elements_x,
                                                          number_detector_elements_y=number_elements_y,
                                                          center_frequency_hz=float(50e6),
                                                          bandwidth_percent=100.0,
                                                          sampling_frequency_mhz=500.0,
                                                          detector_element_width_mm=1,
                                                          detector_element_length_mm=1)

        self.set_detection_geometry(detection_geometry)

        illumination_geometry = PencilArrayIlluminationGeometry(pitch_mm=element_spacing_mm,
                                                                number_illuminators_x=number_elements_x,
                                                                number_illuminators_y=number_elements_y)

        self.add_illumination_geometry(illumination_geometry)


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = RSOMExplorerP50(element_spacing_mm=0.5,
                             number_elements_y=9,
                             number_elements_x=9)
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 12
    settings[Tags.DIM_VOLUME_Y_MM] = 12
    settings[Tags.DIM_VOLUME_Z_MM] = 2.8
    settings[Tags.SPACING_MM] = 0.02
    settings[Tags.STRUCTURES] = {}

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))

    positions = device.detection_geometry.get_detector_element_positions_accounting_for_device_position_mm(settings)
    detector_elements = device.detection_geometry.get_detector_element_orientations(global_settings=settings)
    # detector_elements[:, 1] = detector_elements[:, 1] + device.probe_height_mm
    # positions = np.round(positions / settings[Tags.SPACING_MM]).astype(int)

    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 1], marker='x')
    plt.show()
