"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.core.device_digital_twins import PhotoacousticDevice, \
    CurvedArrayDetectionGeometry, MSOTAcuityIlluminationGeometry
from simpa.utils.settings import Settings
from simpa.utils import Tags
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
import numpy as np


class MSOTAcuityEcho(PhotoacousticDevice):
    """
    This class represents a digital twin of the MSOT Acuity Echo, manufactured by iThera Medical, Munich, Germany
    (https://www.ithera-medical.com/products/msot-acuity/). It is based on the real specifications of the device, but
    due to the limitations of the possibilities how to represent a device in the software frameworks,
    constitutes only an approximation.

    The origin for this device is the center of the membrane at the point of contact between the membrane and the
    tissue, i.e. the outer center of the membrane.

    Some important publications that showcase the use cases of the MSOT Acuity and Acuity Echo device are::

        Regensburger, Adrian P., et al. "Detection of collagens by multispectral optoacoustic
        tomography as an imaging biomarker for Duchenne muscular dystrophy."
        Nature Medicine 25.12 (2019): 1905-1915.

        Knieling, Ferdinand, et al. "Multispectral Optoacoustic Tomography for Assessment of
        Crohn's Disease Activity."
        The New England journal of medicine 376.13 (2017): 1292.

    """

    def __init__(self, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        """

        :param device_position_mm: Outer center of the membrane.
        """
        super(MSOTAcuityEcho, self).__init__(device_position_mm=device_position_mm)

        self.mediprene_membrane_height_mm = 1
        self.probe_height_mm = 43.2
        self.focus_in_field_of_view_mm = 8
        detection_geometry_position_vector = np.add(self.device_position_mm,
                                                    np.array([0, 0,
                                                              self.probe_height_mm + self.focus_in_field_of_view_mm]))

        if field_of_view_extent_mm is None:
            field_of_view_extent_mm = np.asarray([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                  (2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                  0, 0, 0, 50])

        field_of_view_extent_mm[4] -= self.focus_in_field_of_view_mm
        field_of_view_extent_mm[5] -= self.focus_in_field_of_view_mm

        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.34,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.24,
                                                          detector_element_length_mm=13,
                                                          center_frequency_hz=3.96e6,
                                                          bandwidth_percent=55,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=np.pi,
                                                          device_position_mm=detection_geometry_position_vector,
                                                          field_of_view_extent_mm=field_of_view_extent_mm)

        self.set_detection_geometry(detection_geometry)
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION
        illumination_geometry = MSOTAcuityIlluminationGeometry()
        self.add_illumination_geometry(illumination_geometry)

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings: Settings):
        try:
            volume_creator_settings = Settings(global_settings.get_volume_creation_settings())
        except KeyError as e:
            self.logger.warning("You called the update_settings_for_use_of_model_based_volume_creator method "
                                "even though there are no volume creation settings defined in the "
                                "settings dictionary.")
            return

        probe_size_mm = self.probe_height_mm
        mediprene_layer_height_mm = self.mediprene_membrane_height_mm
        heavy_water_layer_height_mm = probe_size_mm - mediprene_layer_height_mm

        new_volume_height_mm = global_settings[Tags.DIM_VOLUME_Z_MM] + mediprene_layer_height_mm + \
                               heavy_water_layer_height_mm

        # adjust the z-dim to msot probe height
        global_settings[Tags.DIM_VOLUME_Z_MM] = new_volume_height_mm

        # adjust the x-dim to msot probe width
        # 1 mm is added (0.5 mm on both sides) to make sure no rounding errors lead to a detector element being outside
        # of the simulated volume.

        if global_settings[Tags.DIM_VOLUME_X_MM] < round(self.detection_geometry.probe_width_mm) + 1:
            width_shift_for_structures_mm = (round(self.detection_geometry.probe_width_mm) + 1 - global_settings[Tags.DIM_VOLUME_X_MM]) / 2
            global_settings[Tags.DIM_VOLUME_X_MM] = round(self.detection_geometry.probe_width_mm) + 1
            self.logger.debug(f"Changed Tags.DIM_VOLUME_X_MM to {global_settings[Tags.DIM_VOLUME_X_MM]}")
        else:
            width_shift_for_structures_mm = 0

        self.logger.debug(volume_creator_settings)

        for structure_key in volume_creator_settings[Tags.STRUCTURES]:
            self.logger.debug("Adjusting " + str(structure_key))
            structure_dict = volume_creator_settings[Tags.STRUCTURES][structure_key]
            if Tags.STRUCTURE_START_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_START_MM][0] = structure_dict[Tags.STRUCTURE_START_MM][
                                                                 0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_START_MM][2] = structure_dict[Tags.STRUCTURE_START_MM][
                                                                 2] + self.probe_height_mm
            if Tags.STRUCTURE_END_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_END_MM][0] = structure_dict[Tags.STRUCTURE_END_MM][
                                                               0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_END_MM][2] = structure_dict[Tags.STRUCTURE_END_MM][
                                                               2] + self.probe_height_mm

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_thickness = np.random.normal(0.4, 0.1)
            us_gel_layer_settings = Settings({
                Tags.PRIORITY: 5,
                Tags.STRUCTURE_START_MM: [0, 0,
                                          heavy_water_layer_height_mm - us_gel_thickness + mediprene_layer_height_mm],
                Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm + mediprene_layer_height_mm],
                Tags.CONSIDER_PARTIAL_VOLUME: True,
                Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.ultrasound_gel(),
                Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
            })

            volume_creator_settings[Tags.STRUCTURES]["us_gel"] = us_gel_layer_settings
        else:
            us_gel_thickness = 0

        mediprene_layer_settings = Settings({
            Tags.PRIORITY: 5,
            Tags.STRUCTURE_START_MM: [0, 0, heavy_water_layer_height_mm - us_gel_thickness],
            Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm - us_gel_thickness + mediprene_layer_height_mm],
            Tags.CONSIDER_PARTIAL_VOLUME: True,
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.mediprene(),
            Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
        })

        volume_creator_settings[Tags.STRUCTURES]["mediprene"] = mediprene_layer_settings

        background_settings = Settings({
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.heavy_water(),
            Tags.STRUCTURE_TYPE: Tags.BACKGROUND
        })
        volume_creator_settings[Tags.STRUCTURES][Tags.BACKGROUND] = background_settings


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = MSOTAcuityEcho()
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 100
    settings[Tags.DIM_VOLUME_Y_MM] = 50
    settings[Tags.DIM_VOLUME_Z_MM] = 100
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}
    # settings[Tags.DIGITAL_DEVICE_POSITION] = [50, 50, 50]
    device.update_settings_for_use_of_model_based_volume_creator(settings)

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))

    positions = device.detection_geometry.get_detector_element_positions_accounting_for_device_position_mm()
    orientations = device.detection_geometry.get_detector_element_orientations(settings)
    # detector_elements[:, 1] = detector_elements[:, 1] + device.probe_height_mm
    # detector_positions = np.round(detector_positions / settings[Tags.SPACING_MM]).astype(int)
    # position_map = np.zeros((x_dim, z_dim))
    # position_map[detector_positions[:, 0], detector_positions[:, 2]] = 1
    middle_point = int(positions.shape[0]/2)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("In Volume")
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], orientations[:, 0], orientations[:, 2])
    fov = device.detection_geometry.get_field_of_view_mm()
    plt.plot([fov[0], fov[1], fov[1], fov[0], fov[0]], [fov[4], fov[4], fov[5], fov[5], fov[4]], color="red")
    plt.subplot(1, 2, 2)
    plt.title("Base")
    positions = device.detection_geometry.get_detector_element_positions_base_mm()
    fov = device.detection_geometry.field_of_view_extent_mm
    plt.plot([fov[0], fov[1], fov[1], fov[0], fov[0]], [fov[4], fov[4], fov[5], fov[5], fov[4]], color="red")
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.show()
    # plt.imshow(map)
    # plt.show()
