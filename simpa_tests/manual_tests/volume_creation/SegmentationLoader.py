# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
from skimage.data import shepp_logan_phantom
from scipy.ndimage import zoom
from simpa_tests.manual_tests import ManualIntegrationTestClass

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SegmentationLoaderTest(ManualIntegrationTestClass):

    def setup(self):
        self.path_manager = sp.PathManager()
        target_spacing = 1.0
        label_mask = shepp_logan_phantom()
        label_mask = np.digitize(label_mask, bins=np.linspace(0.0, 1.0, 11), right=True)
        label_mask = np.reshape(label_mask, (400, 1, 400))
        input_spacing = 0.2
        segmentation_volume_tiled = np.tile(label_mask, (1, 128, 1))
        segmentation_volume_mask = np.round(zoom(segmentation_volume_tiled, input_spacing/target_spacing,
                                                 order=0)).astype(int)

        def segmentation_class_mapping():
            ret_dict = dict()
            ret_dict[0] = sp.TISSUE_LIBRARY.heavy_water()
            ret_dict[1] = sp.TISSUE_LIBRARY.blood()
            ret_dict[2] = sp.TISSUE_LIBRARY.epidermis()
            ret_dict[3] = sp.TISSUE_LIBRARY.muscle()
            ret_dict[4] = sp.TISSUE_LIBRARY.mediprene()
            ret_dict[5] = sp.TISSUE_LIBRARY.ultrasound_gel()
            ret_dict[6] = sp.TISSUE_LIBRARY.heavy_water()
            ret_dict[7] = (sp.MolecularCompositionGenerator()
                           .append(sp.MOLECULE_LIBRARY.oxyhemoglobin(0.01))
                           .append(sp.MOLECULE_LIBRARY.deoxyhemoglobin(0.01))
                           .append(sp.MOLECULE_LIBRARY.water(0.98))
                           .get_molecular_composition(sp.SegmentationClasses.COUPLING_ARTIFACT))
            ret_dict[8] = sp.TISSUE_LIBRARY.heavy_water()
            ret_dict[9] = sp.TISSUE_LIBRARY.heavy_water()
            ret_dict[10] = sp.TISSUE_LIBRARY.heavy_water()
            ret_dict[11] = sp.TISSUE_LIBRARY.heavy_water()
            return ret_dict

        self.settings = sp.Settings()
        self.settings[Tags.SIMULATION_PATH] = self.path_manager.get_hdf5_file_save_path()
        self.settings[Tags.VOLUME_NAME] = "SegmentationTest"
        self.settings[Tags.RANDOM_SEED] = 1234
        self.settings[Tags.WAVELENGTHS] = [700]
        self.settings[Tags.SPACING_MM] = target_spacing
        self.settings[Tags.DIM_VOLUME_X_MM] = 400 / (target_spacing / input_spacing)
        self.settings[Tags.DIM_VOLUME_Y_MM] = 128 / (target_spacing / input_spacing)
        self.settings[Tags.DIM_VOLUME_Z_MM] = 400 / (target_spacing / input_spacing)
        # self.settings[Tags.IGNORE_QA_ASSERTIONS] = True

        self.settings.set_volume_creation_settings({
            Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
            Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),

        })

        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: self.path_manager.get_mcx_binary_path(),
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        })

        self.pipeline = [
            sp.SegmentationBasedVolumeCreationAdapter(self.settings),
            sp.MCXAdapter(self.settings)
        ]

    def perform_test(self):
        sp.simulate(self.pipeline, self.settings, sp.RSOMExplorerP50(element_spacing_mm=2.0,
                                                                     number_elements_y=10,
                                                                     number_elements_x=20,
                                                                     device_position_mm=np.asarray([20, 10, 0])))

    def tear_down(self):
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

    def visualise_result(self, show_figure_on_screen=True, save_path=None):

        if show_figure_on_screen:
            save_path = None
        else:
            save_path = save_path + "SegmentationLoaderExample.png"

        sp.visualise_data(path_to_hdf5_file=self.path_manager.get_hdf5_file_save_path() + "/" + "SegmentationTest" + ".hdf5",
                          wavelength=700,
                          show_initial_pressure=True,
                          show_segmentation_map=True,
                          show_absorption=True,
                          show_fluence=True,
                          show_tissue_density=True,
                          show_speed_of_sound=True,
                          show_anisotropy=True,
                          show_scattering=True,
                          save_path=save_path,
                          log_scale=False)


if __name__ == "__main__":
    test = SegmentationLoaderTest()
    test.run_test(show_figure_on_screen=False)
