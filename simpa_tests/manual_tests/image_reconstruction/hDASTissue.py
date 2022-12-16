# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation import simulate
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils import Tags, Settings, TISSUE_LIBRARY
from simpa.utils.path_manager import PathManager
from simpa import KWaveAdapter, DelayAndSumAdapter, visualise_device
from simpa.core.device_digital_twins import *
from simpa.io_handling import  load_data_field
import numpy as np
import matplotlib.pyplot as plt
import os
from simpa_tests.manual_tests import ManualIntegrationTestClass
from simpa.core.device_digital_twins import SlitIlluminationGeometry, LinearArrayDetectionGeometry, PhotoacousticDevice
from simpa import KWaveAdapter, MCXAdapter, \
    DelayAndSumAdapter, ModelBasedVolumeCreationAdapter, GaussianNoise

from simpa_tests.manual_tests import ManualIntegrationTestClass

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class hDASTissue(ManualIntegrationTestClass):
    """
    This test runs a simulation creating an example volume of geometric shapes and reconstructs it with the Delay and
    Sum algorithm assuming a fixed SoS-value (DAS) and also using the heterogeneous SoS-map (hDAS).
    To verify that the test was successful a user has to evaluate the displayed reconstruction.
    """ 
    
    def __init__(self, device_type="Linear"):
        self.device_type = device_type

    def setup(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        Please make sure that a valid path_config.env file is located in your home directory, or that you
        point to the correct file in the PathManager().
        """

        self.path_manager = PathManager()
        self.VOLUME_TRANSDUCER_DIM_IN_MM = 75
        self.VOLUME_PLANAR_DIM_IN_MM = 20
        self.VOLUME_HEIGHT_IN_MM = 25
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711
        np.random.seed(self.RANDOM_SEED)
        print(f"Use {self.device_type} Detector")
        if self.device_type == "Acuity":
            self.device = MSOTAcuityEcho(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                  self.VOLUME_PLANAR_DIM_IN_MM/2, 0]))
        elif self.device_type == "Linear":
            self.device = PhotoacousticDevice(
                device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                             self.VOLUME_PLANAR_DIM_IN_MM/2, 0]),
                field_of_view_extent_mm=np.asarray([-self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     0, 0, 0, 2*self.VOLUME_HEIGHT_IN_MM])
                )
            self.device.set_detection_geometry(LinearArrayDetectionGeometry(
                                                device_position_mm=self.device.device_position_mm,
                                                number_detector_elements=200,
                                                pitch_mm = 0.25,
                                                #field_of_view_extent_mm=self.device.field_of_view_extent_mm
                                                )
                                                )
            self.device.add_illumination_geometry(SlitIlluminationGeometry(slit_vector_mm=[50,0,0]))
        else:
            raise("not supplemented detector chosen")
        
        self.general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: "TestImageReconstruction_" + str(self.RANDOM_SEED),
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,

            # Simulation Device
            Tags.DIGITAL_DEVICE: Tags.DIGITAL_DEVICE_MSOT_ACUITY,

            # The following parameters set the optical forward model
            Tags.WAVELENGTHS: [700]
        }
        self.settings = Settings(self.general_settings)

        self.settings.set_volume_creation_settings({
            Tags.STRUCTURES: self.create_example_tissue(),
            Tags.SIMULATE_DEFORMED_LAYERS: False,
            Tags.US_GEL: False
        })

        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: self.path_manager.get_mcx_binary_path(),
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        })

        self.settings.set_acoustic_settings({
            Tags.ACOUSTIC_SIMULATION_3D: True,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
            Tags.GPU: True,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.05,
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True
        })

        self.settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
            Tags.TUKEY_WINDOW_ALPHA: 0.5,
            Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
            Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e6),
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
            Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_DIFFERENTIAL,
            Tags.SPACING_MM: self.settings[Tags.SPACING_MM],
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.ACOUSTIC_SIMULATION_3D: False,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True,
            #Tags.DATA_FIELD_SPEED_OF_SOUND: 1540, # if commented, then use mean sos value for homogenous recon
            Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
            Tags.DATA_FIELD_DENSITY: 1000,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
        })

    def create_example_tissue(self):
        """
        4 circular shaped vessels with increasing radius, no deformation
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.subcutaneous_fat()
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        muscle_dictionary = Settings()
        muscle_dictionary[Tags.PRIORITY] = 1
        muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 20]
        muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 50]
        muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        muscle_dictionary[Tags.SIMULATE_DEFORMED_LAYERS] = False
        muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
        muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        fat_dictionary = Settings()
        fat_dictionary[Tags.PRIORITY] = 1
        fat_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        fat_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 20]
        fat_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.subcutaneous_fat()
        fat_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        fat_dictionary[Tags.SIMULATE_DEFORMED_LAYERS] = False
        fat_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
        fat_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        vessel_1_dictionary = Settings()
        vessel_1_dictionary[Tags.PRIORITY] = 3
        vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 9,
                                                        0, 15]
        vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 9,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 0.1
        vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_2_dictionary = Settings()
        vessel_2_dictionary[Tags.PRIORITY] = 3
        vessel_2_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 3,
                                                        0, 15]
        vessel_2_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 3,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 0.5
        vessel_2_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_3_dictionary = Settings()
        vessel_3_dictionary[Tags.PRIORITY] = 3
        vessel_3_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 3,
                                                        0, 15]
        vessel_3_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 3,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_3_dictionary[Tags.STRUCTURE_RADIUS_MM] = 1.
        vessel_3_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_3_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_3_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_4_dictionary = Settings()
        vessel_4_dictionary[Tags.PRIORITY] = 3
        vessel_4_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 9,
                                                        0, 15]
        vessel_4_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 9,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_4_dictionary[Tags.STRUCTURE_RADIUS_MM] = 1.5
        vessel_4_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_4_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_4_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["muscle"] = muscle_dictionary
        tissue_dict["fat"] = fat_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        tissue_dict["vessel_2"] = vessel_2_dictionary
        tissue_dict["vessel_3"] = vessel_3_dictionary
        tissue_dict["vessel_4"] = vessel_4_dictionary
        return tissue_dict

    def perform_test(self):
        """
        Simulate and reconstruct with DAS and hDAS
        """

        self.device.update_settings_for_use_of_model_based_volume_creator(self.settings)

        SIMUATION_PIPELINE = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
            #GaussianNoise(self.settings, "noise_initial_pressure"),
            KWaveAdapter(self.settings),
            DelayAndSumAdapter(self.settings)
        ]

        simulate(SIMUATION_PIPELINE, self.settings, self.device)

        self.initial_pressure = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_INITIAL_PRESSURE,
                                                self.settings[Tags.WAVELENGTH])

        self.reconstructed_image_homo = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                        self.settings[Tags.WAVELENGTH])
        self.speed_of_sound_map_hetero = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_SPEED_OF_SOUND)

        if Tags.DATA_FIELD_SPEED_OF_SOUND in self.settings.get_reconstruction_settings():
            self.homogenous_speed_of_sound = self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND]
        else:
            self.homogenous_speed_of_sound = np.mean(self.speed_of_sound_map_hetero)

        # heterogeneous Reconstruction
        self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND] = self.speed_of_sound_map_hetero
        self.settings[Tags.SOS_HETEROGENEOUS] = True
        DelayAndSumAdapter(self.settings).run(self.device)

        self.reconstructed_image_hetero = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                          self.settings[Tags.WAVELENGTH])

    # overwrite function of ReconstructionAlgorithmTestBaseClass
    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        """
        plot sos, p0, hDAS and DAS reconstruction (zoomed and not zoomed) and the difference
        """

        # visualize detector
        #visualise_device(self.pa_device)

        slice = 20

        fig, axes = plt.subplots(2, 2)
        fig.canvas.manager.set_window_title("hDAS in Comparison to DAS")
        axes[0,0].set_title(f"Simulated initial pressure")
        im = axes[0,0].imshow(np.rot90(self.initial_pressure[:, slice, :], 3), cmap="gnuplot")
        plt.colorbar(im, ax=axes[0,0])
   

        axes[0,1].set_title(f"""Hetero SoS map\nmean = {
            self.speed_of_sound_map_hetero[:,slice,:].mean():.1f}""")
        im = axes[0,1].imshow(np.rot90(self.speed_of_sound_map_hetero[:, slice, :], 3))
        plt.colorbar(im, ax=axes[0,1])

        axes[1,0].set_title(f"DAS Reconstruction\nwith SoS={self.homogenous_speed_of_sound:.1f}")
        im = axes[1,0].imshow(np.rot90(self.reconstructed_image_homo, 3), cmap="gnuplot")
        plt.colorbar(im, ax=axes[1,0])

        axes[1,1].set_title(f"hDAS Reconstruction")
        im = axes[1,1].imshow(np.rot90(self.reconstructed_image_hetero, 3), cmap="gnuplot")
        plt.colorbar(im, ax=axes[1,1])

        for ax in axes.flatten():
            ax.set_xlabel("x")
            ax.set_ylabel("z")

        plt.tight_layout()

        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = "figures/hDAS/"
            plt.savefig(save_path + f"hDAS_full_pipeline_{self.device_type}.png", dpi=300)
        plt.close()

        fig, axes = plt.subplots(3,4)
        fig.canvas.manager.set_window_title("hDAS/hDAS in Comparison to DAS (Zoomed)")
        # get slices for the vessels
        edges = np.array([[48, 68], [53, 73]]) if self.device_type == "Linear" else np.array([[48, 68], [95, 115]])
        edges_pzero = np.array([[50, 70], [104, 124]]) if self.device_type == "Linear" else np.array([[223, 243], [104, 124]])
        for i in range(4):
            axes[0,i].imshow(np.rot90(self.reconstructed_image_homo, 3)[edges[0,0]:edges[0,1], edges[1,0]:edges[1,1]], cmap="gnuplot")
            axes[0,i].set_xticks([])
            axes[0,i].set_yticks([])
            axes[1,i].imshow(np.rot90(self.reconstructed_image_hetero, 3)[edges[0,0]:edges[0,1], edges[1,0]:edges[1,1]], cmap="gnuplot")
            axes[1,i].set_xticks([])
            axes[1,i].set_yticks([])
            edges[1,:] += 24
            axes[2,i].imshow(np.rot90(self.initial_pressure[:, slice, :], 3)[edges_pzero[0,0]:edges_pzero[0,1], edges_pzero[1,0]:edges_pzero[1,1]],
                             cmap="gnuplot")
            axes[2,i].set_xticks([])
            axes[2,i].set_yticks([])
            edges_pzero[1,:] += 24
        axes[0,0].set_ylabel('DAS', rotation=0, fontsize=20, labelpad=50)
        axes[1,0].set_ylabel('hDAS', rotation=0, fontsize=20, labelpad=50)
        axes[2,0].set_ylabel('p0', rotation=0, fontsize=20, labelpad=50)
        plt.tight_layout()
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"hDAS_full_pipeline_zoomed_{self.device_type}.png", dpi=300)
        plt.close()
        

        plt.figure("Difference of hDAS and DAS")
        plt.title(f"hDAS - DAS")
        difference = np.rot90(self.reconstructed_image_hetero, 3)-np.rot90(self.reconstructed_image_homo, 3)
        max_difference = np.abs(difference).max()
        plt.imshow(difference, cmap = "seismic", vmin = -max_difference, vmax = max_difference)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        plt.tight_layout()
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"hDAS_full_pipeline_difference_{self.device_type}.png", dpi=300)
        plt.close()
    

if __name__ == '__main__':
    test = hDASTissue(device_type="Acuity") # "Acuity" or "Linear"
    test.run_test(show_figure_on_screen=False)
