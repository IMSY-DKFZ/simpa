# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

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
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of e.g. detector element positions or illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(MSOTAcuityEcho, self).__init__(device_position_mm=device_position_mm)

        self.mediprene_membrane_height_mm = 1
        self.probe_height_mm = 43.2
        self.focus_in_field_of_view_mm = 8
        self.detection_geometry_position_vector = np.add(self.device_position_mm,
                                                         np.array([0, 0,
                                                                   self.probe_height_mm +
                                                                   self.focus_in_field_of_view_mm]))

        if field_of_view_extent_mm is None:
            self.field_of_view_extent_mm = np.asarray([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                       (2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                       0, 0, 0, 50])
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

        self.field_of_view_extent_mm[4] -= self.focus_in_field_of_view_mm
        self.field_of_view_extent_mm[5] -= self.focus_in_field_of_view_mm

        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.34,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.24,
                                                          detector_element_length_mm=13,
                                                          center_frequency_hz=3.96e6,
                                                          bandwidth_percent=55,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=np.pi,
                                                          device_position_mm=self.detection_geometry_position_vector,
                                                          field_of_view_extent_mm=self.field_of_view_extent_mm)

        self.set_detection_geometry(detection_geometry)
        illumination_geometry = MSOTAcuityIlluminationGeometry()

        # y position relative to the membrane:
        # The laser is located 43.2 mm  behind the membrane with an angle of 22.4 degrees.
        # However, the incident of laser and image plane is located 2.8 behind the membrane (outside of the device).
        y_pos_relative_to_membrane = np.tan(np.deg2rad(22.4)) * (43.2 + 2.8)
        self.add_illumination_geometry(illumination_geometry,
                                       illuminator_position_relative_to_pa_device=np.array([0,
                                                                                            -y_pos_relative_to_membrane,
                                                                                            -43.2]))

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings):
        """
        Updates the volume creation settings of the model based volume creator according to the size of the device.
        :param global_settings: Settings for the entire simulation pipeline.
        :type global_settings: Settings
        """
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

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_thickness = np.random.normal(0.4, 0.1)
        else:
            us_gel_thickness = 0

        z_dim_position_shift_mm = mediprene_layer_height_mm + heavy_water_layer_height_mm + us_gel_thickness

        new_volume_height_mm = global_settings[Tags.DIM_VOLUME_Z_MM] + z_dim_position_shift_mm

        # adjust the z-dim to msot probe height
        global_settings[Tags.DIM_VOLUME_Z_MM] = new_volume_height_mm

        # adjust the x-dim to msot probe width
        # 1 mm is added (0.5 mm on both sides) to make sure no rounding errors lead to a detector element being outside
        # of the simulated volume.

        if global_settings[Tags.DIM_VOLUME_X_MM] < round(self.detection_geometry.probe_width_mm) + 1:
            width_shift_for_structures_mm = (round(self.detection_geometry.probe_width_mm) + 1 -
                                             global_settings[Tags.DIM_VOLUME_X_MM]) / 2
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
                                                                 2] + z_dim_position_shift_mm
            if Tags.STRUCTURE_END_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_END_MM][0] = structure_dict[Tags.STRUCTURE_END_MM][
                                                               0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_END_MM][2] = structure_dict[Tags.STRUCTURE_END_MM][
                                                               2] + z_dim_position_shift_mm

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_layer_settings = Settings({
                Tags.PRIORITY: 5,
                Tags.STRUCTURE_START_MM: [0, 0,
                                          heavy_water_layer_height_mm + mediprene_layer_height_mm],
                Tags.STRUCTURE_END_MM: [0, 0,
                                        heavy_water_layer_height_mm + mediprene_layer_height_mm + us_gel_thickness],
                Tags.CONSIDER_PARTIAL_VOLUME: True,
                Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.ultrasound_gel(),
                Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
            })

            volume_creator_settings[Tags.STRUCTURES]["us_gel"] = us_gel_layer_settings

        mediprene_layer_settings = Settings({
            Tags.PRIORITY: 5,
            Tags.STRUCTURE_START_MM: [0, 0, heavy_water_layer_height_mm],
            Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm + mediprene_layer_height_mm],
            Tags.CONSIDER_PARTIAL_VOLUME: True,
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.mediprene(),
            Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
        })

        volume_creator_settings[Tags.STRUCTURES]["mediprene"] = mediprene_layer_settings

        self.device_position_mm = np.add(self.device_position_mm, np.array([width_shift_for_structures_mm, 0,
                                                                            probe_size_mm]))
        self.detection_geometry_position_vector = np.add(self.device_position_mm,
                                                         np.array([0, 0,
                                                                   self.focus_in_field_of_view_mm]))
        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.34,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.24,
                                                          detector_element_length_mm=13,
                                                          center_frequency_hz=3.96e6,
                                                          bandwidth_percent=55,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=np.pi,
                                                          device_position_mm=self.detection_geometry_position_vector,
                                                          field_of_view_extent_mm=self.field_of_view_extent_mm)

        self.set_detection_geometry(detection_geometry)
        for illumination_geom in self.illumination_geometries:
            illumination_geom.device_position_mm = np.add(illumination_geom.device_position_mm,
                                                          np.array([0, 0, probe_size_mm]))

        background_settings = Settings({
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.heavy_water(),
            Tags.STRUCTURE_TYPE: Tags.BACKGROUND
        })
        volume_creator_settings[Tags.STRUCTURES][Tags.BACKGROUND] = background_settings

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"MSOTAcuityEcho": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = MSOTAcuityEcho()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
