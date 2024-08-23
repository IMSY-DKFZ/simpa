# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

# FIXME temporary workaround for newest Intel architectures
import os

import numpy as np

from simpa import (DelayAndSumAdapter, FieldOfViewCropping, KWaveAdapter,
                   MCXAdapter, ModelBasedAdapter)
from simpa.core.device_digital_twins import *
from simpa.core.simulation import simulate
from simpa.io_handling import load_data_field
from simpa.utils import Tags
from simpa.utils.libraries.molecule_library import (
    MolecularCompositionGenerator, Molecule)
from simpa.utils.libraries.spectrum_library import (AbsorptionSpectrumLibrary,
                                                    AnisotropySpectrumLibrary,
                                                    ScatteringSpectrumLibrary)
from simpa.utils.path_manager import PathManager
from simpa.utils.settings import Settings
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from simpa_tests.manual_tests import ReconstructionAlgorithmTestBaseClass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PointSourceReconstruction(ReconstructionAlgorithmTestBaseClass):
    """
    TODO

    """
    def __init__(self, speed_of_sound: float = 1470, volume_transducer_dim_in_mm: float = 90,
                 volume_planar_dim_in_mm: float = 20, volume_height_in_mm: float = 90,
                 spacing: float = 0.4):
        
        self.reconstructed_image_pipeline = None # TODO REMOVE

        self.SPEED_OF_SOUND = speed_of_sound
        self.VOLUME_TRANSDUCER_DIM_IN_MM = volume_transducer_dim_in_mm
        self.VOLUME_PLANAR_DIM_IN_MM = volume_planar_dim_in_mm
        self.VOLUME_HEIGHT_IN_MM = volume_height_in_mm
        self.SPACING = spacing

        self.RANDOM_SEED = 4711

    def create_point_source(self):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        background_molecule = Molecule(
            volume_fraction=1.0,
            absorption_spectrum=AbsorptionSpectrumLibrary.CONSTANT_ABSORBER_ARBITRARY(1e-10),
            scattering_spectrum=ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(1e-10),
            anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(1.0),
            speed_of_sound=self.SPEED_OF_SOUND
        )
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = (MolecularCompositionGenerator().
                                                            append(background_molecule).
                                                            get_molecular_composition(-1))
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        vessel_molecule = Molecule(
            volume_fraction=1.0,
            absorption_spectrum=AbsorptionSpectrumLibrary.CONSTANT_ABSORBER_ARBITRARY(2),
            scattering_spectrum=ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(100),
            anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(0.9),
            speed_of_sound=self.SPEED_OF_SOUND+100,
            density=1100
        )

        vessel_1_dictionary = Settings()
        vessel_1_dictionary[Tags.PRIORITY] = 3
        vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM/2-10, 0, 35]
        vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM/2-10, self.VOLUME_PLANAR_DIM_IN_MM, 35]
        vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = self.SPACING
        vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = (MolecularCompositionGenerator().
                                                        append(vessel_molecule).
                                                        get_molecular_composition(-1))
        vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        return tissue_dict

    def setup(self):

        # Note: Please make sure that a valid path_config.env file is located in your home directory, or that you
        #  point to the correct file in the PathManager().
        self.path_manager = PathManager()

    
        # Seed the numpy random configuration prior to creating the global_settings file in
        # order to ensure that the same volume
        # is generated with the same random seed every time.

        np.random.seed(self.RANDOM_SEED)
        self.VOLUME_NAME = "CompletePipelineTestMSOT_"+str(self.RANDOM_SEED)

        general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: self.VOLUME_NAME,
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,

            # The following parameters set the optical forward model
            Tags.WAVELENGTHS: [700]
        }
        settings = Settings(general_settings)
        np.random.seed(self.RANDOM_SEED)

        settings.set_volume_creation_settings({
            Tags.STRUCTURES: self.create_point_source(),
            Tags.SIMULATE_DEFORMED_LAYERS: True
        })

        settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: self.path_manager.get_mcx_binary_path(),
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        })

        settings.set_acoustic_settings({
            Tags.ACOUSTIC_SIMULATION_3D: False,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True
        })

        settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
            Tags.TUKEY_WINDOW_ALPHA: 0.5,
            Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(8e6),
            Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(0.1e4),
            Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True,
            Tags.DATA_FIELD_SPEED_OF_SOUND: self.SPEED_OF_SOUND,
            Tags.SPACING_MM: self.SPACING
        })

        self.settings = settings

    def simulate_and_evaluate_with_device(self, _device):
            SIMULATION_PIPELINE = [
                ModelBasedAdapter(self.settings),
                MCXAdapter(self.settings),
                KWaveAdapter(self.settings),
                FieldOfViewCropping(self.settings),
                DelayAndSumAdapter(self.settings)
            ]

            print("Simulating for device:", _device)
            simulate(SIMULATION_PIPELINE, self.settings, _device)

            if Tags.WAVELENGTH in self.settings:
                wavelength = self.settings[Tags.WAVELENGTH]
            else:
                wavelength = 700

            initial_pressure = load_data_field(self.path_manager.get_hdf5_file_save_path() + "/" + self.VOLUME_NAME + ".hdf5",
                                            data_field=Tags.DATA_FIELD_INITIAL_PRESSURE,
                                            wavelength=wavelength)
            reconstruction = load_data_field(self.path_manager.get_hdf5_file_save_path() + "/" + self.VOLUME_NAME + ".hdf5",
                                            data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                            wavelength=wavelength)

            p0_idx = np.unravel_index(np.argmax(initial_pressure), np.shape(initial_pressure))
            re_idx = np.unravel_index(np.argmax(reconstruction), np.shape(reconstruction))

            print("x/y in initial pressure map:", p0_idx)
            print("x/y in reconstruction map:", re_idx)
            distance = np.sqrt((re_idx[0] - p0_idx[0]) ** 2 + (re_idx[1] - p0_idx[1]) ** 2)
            print("Distance:", distance)

            if self.save_path is not None:
                save_path = self.save_path + f"PointSourceReconstruction_{self.figure_number}.png"
            else:
                save_path = self.save_path

            visualise_data(path_to_hdf5_file=self.path_manager.get_hdf5_file_save_path() + "/" + self.VOLUME_NAME + ".hdf5",
                        wavelength=wavelength,
                        show_time_series_data=True,
                        show_absorption=False,
                        show_reconstructed_data=True,
                        show_xz_only=True,
                        show_initial_pressure=True,
                        show_segmentation_map=False,
                        log_scale=False,
                        save_path=save_path)
            self.figure_number += 1
            return distance
    

    def test_reconstruction_of_simulation(self):     


        dist = list()

        # fov_e = np.asarray([-self.VOLUME_TRANSDUCER_DIM_IN_MM/2, self.VOLUME_TRANSDUCER_DIM_IN_MM/2, 0, 0, 0, self.VOLUME_HEIGHT_IN_MM])
        # device = PhotoacousticDevice(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
        #                                                           self.VOLUME_PLANAR_DIM_IN_MM/2,
        #                                                           0]),
        #                              field_of_view_extent_mm=fov_e)
        # device.set_detection_geometry(Random2DArrayDetectionGeometry(device_position_mm=device.device_position_mm,
        #                                                              number_detector_elements=256,
        #                                                              seed=1234, field_of_view_extent_mm=fov_e))
        # device.add_illumination_geometry(PencilBeamIlluminationGeometry())
        # dist.append(self.simulate_and_evaluate_with_device(device))
        #
        # device = PhotoacousticDevice(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
        #                                                           self.VOLUME_PLANAR_DIM_IN_MM/2,
        #                                                           0]),
        #                              field_of_view_extent_mm=fov_e)
        # device.set_detection_geometry(Random3DArrayDetectionGeometry(device_position_mm=device.device_position_mm,
        #                                                              number_detector_elements=256,
        #                                                              seed=1234, field_of_view_extent_mm=fov_e))
        # device.add_illumination_geometry(PencilBeamIlluminationGeometry())
        # dist.append(self.simulate_and_evaluate_with_device(device))
        
        dist.append(self.simulate_and_evaluate_with_device(MSOTAcuityEcho(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                                                self.VOLUME_PLANAR_DIM_IN_MM/2,
                                                                                                35]),
                                                                    field_of_view_extent_mm=np.array([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                                                                    (2 * np.sin(0.34 /
                                                                                                        40 * 128) * 40) / 2,
                                                                                                    0, 0, -25, 25]))))

        dist.append(self.simulate_and_evaluate_with_device(InVision256TF(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                                                self.VOLUME_PLANAR_DIM_IN_MM/2,
                                                                                                self.VOLUME_HEIGHT_IN_MM/2]))))
        device = PhotoacousticDevice(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                self.VOLUME_PLANAR_DIM_IN_MM/2,
                                                                30]),
                                    field_of_view_extent_mm=np.asarray([-self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                        self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                        0, 0, 0, self.VOLUME_HEIGHT_IN_MM]))
        device.set_detection_geometry(LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                                pitch_mm=0.2,
                                                                number_detector_elements=256))
        device.add_illumination_geometry(PencilBeamIlluminationGeometry(device_position_mm=device.device_position_mm))
        dist.append(self.simulate_and_evaluate_with_device(device))

        device = PhotoacousticDevice(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                self.VOLUME_PLANAR_DIM_IN_MM/2,
                                                                5]),
                                    field_of_view_extent_mm=np.asarray([-self.VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                        self.VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                        0, 0, 0, self.VOLUME_HEIGHT_IN_MM]))

        device.set_detection_geometry(LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                                pitch_mm=0.2,
                                                                number_detector_elements=256))
        device.add_illumination_geometry(PencilBeamIlluminationGeometry())
        dist.append(self.simulate_and_evaluate_with_device(device))

        device = PhotoacousticDevice(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                self.VOLUME_PLANAR_DIM_IN_MM/2,
                                                                10]),
                                    field_of_view_extent_mm=np.asarray([-self.VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                        self.VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                        0, 0, 0, self.VOLUME_HEIGHT_IN_MM]))
        device.set_detection_geometry(LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                                pitch_mm=0.2,
                                                                number_detector_elements=256))
        device.add_illumination_geometry(PencilBeamIlluminationGeometry())
        dist.append(self.simulate_and_evaluate_with_device(device))
        print("")
        print("Results:")
        print("______________")
        for dis in dist:
            print("Distance", dis)

    def run_test(self, show_figure_on_screen=True, save_path=None):

        if save_path is None or not os.path.isdir(save_path):
            save_path = "figures/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if show_figure_on_screen:
            save_path = None

        self.save_path = save_path
        self.figure_number = 0

        self.setup()
        self.perform_test()
        self.tear_down()

if __name__ == '__main__':
    test = PointSourceReconstruction()
    test.run_test(show_figure_on_screen=True)
