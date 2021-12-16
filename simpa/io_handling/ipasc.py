# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import numpy as np
import os
import uuid

from simpa.log import Logger
from simpa.io_handling import load_data_field, load_hdf5
from simpa.core.device_digital_twins import DigitalDeviceTwinBase, PhotoacousticDevice
from simpa.utils import Settings, Tags

from ipasc_tool import BaseAdapter, MetaDatum, DeviceMetaDataCreator, DetectionElementCreator, MetadataAcquisitionTags
from ipasc_tool.iohandler import write_data as write_ipasc_data


class IpascSimpaAdapter(BaseAdapter):
    """
    This class contains the logic to extract the needed meta data from the SIMPA simulation files to
    fulfill the requirements of the IPASC standard data format.
    """

    def __init__(self, hdf5_file_path: str, device: DigitalDeviceTwinBase, settings: Settings = None):
        self.logger = Logger()

        # Input validation with descriptive error messages
        if not os.path.exists(hdf5_file_path):
            self.logger.error(f"The given file path ({hdf5_file_path}) does not exist.")
            raise AssertionError(f"The given file path ({hdf5_file_path}) does not exist.")
        if not os.path.isfile(hdf5_file_path):
            self.logger.error(f"The given file path ({hdf5_file_path}) does not point to a file.")
            raise AssertionError(f"The given file path ({hdf5_file_path}) does not point to a file.")
        if not hdf5_file_path.endswith(".hdf5"):
            self.logger.error(f"The given file path must point to an hdf5 file that ends with '.hdf5'")
            raise AssertionError(f"The given file path must point to an hdf5 file that ends with '.hdf5'")
        self.simpa_hdf5_file_path = hdf5_file_path
        self.ipasc_hdf5_file_path = hdf5_file_path.replace(".hdf5", "_ipasc.hdf5")

        # checking SIMPA settings dictionary
        if settings is None:
            settings = load_hdf5(hdf5_file_path)
            if Tags.SETTINGS not in settings:
                self.logger.error("Unable to recover settings dictionary. Please supply a valid settings dictionary for a "
                                  "successful export.")
            settings = settings[Tags.SETTINGS]
        if settings is None or not isinstance(settings, Settings):
            self.logger.error("No settings found at Tags.SETTINGS in the loaded HDF5 file. "
                              "Please supply a valid settings dictionary for a successful export.")
        self.settings = settings

        # checking given photoacoustic device
        if device is None or not isinstance(device, PhotoacousticDevice):
            self.logger.error("Given device was not a photoacoustic device.")
            raise AssertionError("Given device was not a photoacoustic device.")
        self.device = device

        if Tags.WAVELENGTHS not in settings:
            self.logger.error("Tags.WAVELENGTHS was not defined in the settings dictionary. Aborting IPASC file export.")

        self.wavelengths = self.settings[Tags.WAVELENGTHS]

        # Load the data for the first wavelength just to get the number of elements and number of time steps
        try:
            num_elements, num_time_steps = np.shape(load_data_field(self.simpa_hdf5_file_path,
                                                                    Tags.DATA_FIELD_TIME_SERIES_DATA, self.wavelengths[0]))
        except KeyError as e:
            self.logger.error(e)
            raise AssertionError(e)

        self.time_series_data = np.zeros(shape=(num_elements, num_time_steps, len(self.wavelengths), 1))

        for wl_idx, wavelength in enumerate(self.wavelengths):
            self.time_series_data[:, :, wl_idx, 0] = load_data_field(self.simpa_hdf5_file_path,
                                                                     Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)

        self.time_series_data = self.time_series_data.astype(np.float32)

        super(IpascSimpaAdapter, self).__init__()

    def generate_binary_data(self) -> np.ndarray:
        return self.time_series_data

    def generate_meta_data_device(self) -> dict:
        device_creator = DeviceMetaDataCreator()
        device_creator.set_general_information(uuid=self.device.generate_uuid(),
                                               fov=self.device.field_of_view_extent_mm/1000)

        positions = self.device.get_detection_geometry().get_detector_element_positions_base_mm()/1000
        orientations = self.device.get_detection_geometry().get_detector_element_orientations()

        for idx, (position, orientation) in enumerate(zip(positions, orientations)):
            detection_element_creator = DetectionElementCreator()
            # do not forget to convert to m
            detection_element_creator.set_detector_position(position)
            detection_element_creator.set_detector_orientation(orientation)
            detection_element_creator.set_detector_geometry_type("CUBOID")
            # do not forget to convert to m
            detection_element_creator.set_detector_geometry(
                np.asarray([self.device.get_detection_geometry().detector_element_width_mm,
                            self.device.get_detection_geometry().detector_element_length_mm, 0.0001]) / 1000)
            device_creator.add_detection_element(detection_element_creator.get_dictionary())

        return device_creator.finalize_device_meta_data()

    def set_metadata_value(self, metadata_tag: MetaDatum) -> object:
        if metadata_tag == MetadataAcquisitionTags.UUID:
            return str(uuid.uuid4())
        elif metadata_tag == MetadataAcquisitionTags.DATA_TYPE:
            return str(type(self.time_series_data[0, 0, 0, 0].item()))
        elif metadata_tag == MetadataAcquisitionTags.AD_SAMPLING_RATE:
            if Tags.K_WAVE_SPECIFIC_DT in self.settings and self.settings[Tags.K_WAVE_SPECIFIC_DT]:
                return float(1.0 / self.settings[Tags.K_WAVE_SPECIFIC_DT])
            elif self.device.get_detection_geometry().sampling_frequency_MHz is not None:
                return float(self.device.get_detection_geometry().sampling_frequency_MHz * 1000000)
        elif metadata_tag == MetadataAcquisitionTags.ACQUISITION_OPTICAL_WAVELENGTHS:
            return np.asarray(self.wavelengths)
        elif metadata_tag == MetadataAcquisitionTags.DIMENSIONALITY:
            return "time"
        elif metadata_tag == MetadataAcquisitionTags.SCANNING_METHOD:
            return "full_scan"
        elif metadata_tag == MetadataAcquisitionTags.PHOTOACOUSTIC_IMAGING_DEVICE:
            return str(self.device.generate_uuid())
        elif metadata_tag == MetadataAcquisitionTags.SIZES:
            return np.asarray(np.shape(self.time_series_data))
        else:
            return None


def export_to_ipasc(hdf5_file_path: str, device: DigitalDeviceTwinBase, settings: Settings = None):
    """
    This function exports parts of the SIMPA simulation results of the given hdf5_file_path into the IPASC
    standard data format.


    :param hdf5_file_path: A string with the path to an HDF5 file containing a SIMPA simulation result
    :param device: A PhotoacousticDevice that describes the digital device twin
    :param settings: The settings dictionary used for the simulation. if not given, it is attempted to recover the
                     settings dictionary from the HDF5 file.
    :return: None
    """
    try:
        ipasc_adapter = IpascSimpaAdapter(hdf5_file_path, device, settings)
        pa_data = ipasc_adapter.generate_pa_data()
        write_ipasc_data(ipasc_adapter.ipasc_hdf5_file_path, pa_data)
    except Exception as e:
        Logger().error(e)
