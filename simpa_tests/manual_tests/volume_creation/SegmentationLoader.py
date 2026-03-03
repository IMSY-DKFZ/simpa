# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
from skimage.data import shepp_logan_phantom
from scipy.ndimage import zoom, uniform_filter
from simpa_tests.manual_tests import ManualIntegrationTestClass

# FIXME temporary workaround for newest Intel architectures
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SegmentationLoaderRSOMExplorerTest(ManualIntegrationTestClass):

    def setup(self):
        self.path_manager = sp.PathManager()
        self.volume_name = "SegmentationRSOMExplorerTest"
        target_spacing = 1.0
        label_mask = shepp_logan_phantom()
        label_mask = np.digitize(label_mask, bins=np.linspace(0.0, 1.0, 11), right=True)
        label_mask = np.reshape(label_mask, (400, 1, 400))
        input_spacing = 0.2
        segmentation_volume_tiled = np.tile(label_mask, (1, 128, 1))
        segmentation_volume_mask = sp.round_x5_away_from_zero(
            zoom(segmentation_volume_tiled, input_spacing / target_spacing, order=0)
        ).astype(int)

        def segmentation_class_mapping():
            ret_dict = dict()
            ret_dict[0] = sp.TissueLibrary.heavy_water()
            ret_dict[1] = sp.TissueLibrary.blood()
            ret_dict[2] = sp.TissueLibrary.epidermis()
            ret_dict[3] = sp.TissueLibrary.muscle()
            ret_dict[4] = sp.TissueLibrary.mediprene()
            ret_dict[5] = sp.TissueLibrary.ultrasound_gel()
            ret_dict[6] = sp.TissueLibrary.heavy_water()
            ret_dict[7] = (
                sp.MolecularCompositionGenerator()
                .append(sp.MoleculeLibrary.oxyhemoglobin(0.01))
                .append(sp.MoleculeLibrary.deoxyhemoglobin(0.01))
                .append(sp.MoleculeLibrary.water(0.98))
                .get_molecular_composition(sp.SegmentationClasses.COUPLING_ARTIFACT)
            )
            ret_dict[8] = sp.TissueLibrary.heavy_water()
            ret_dict[9] = sp.TissueLibrary.heavy_water()
            ret_dict[10] = sp.TissueLibrary.heavy_water()
            ret_dict[11] = sp.TissueLibrary.heavy_water()
            return ret_dict

        self.settings = sp.Settings()
        self.settings[Tags.SIMULATION_PATH] = (
            self.path_manager.get_hdf5_file_save_path()
        )
        self.settings[Tags.VOLUME_NAME] = self.volume_name
        self.settings[Tags.RANDOM_SEED] = 1234
        self.settings[Tags.WAVELENGTHS] = [700]
        self.settings[Tags.SPACING_MM] = target_spacing
        self.settings[Tags.DIM_VOLUME_X_MM] = 400 / (target_spacing / input_spacing)
        self.settings[Tags.DIM_VOLUME_Y_MM] = 128 / (target_spacing / input_spacing)
        self.settings[Tags.DIM_VOLUME_Z_MM] = 400 / (target_spacing / input_spacing)
        # self.settings[Tags.IGNORE_QA_ASSERTIONS] = True

        self.settings.set_volume_creation_settings(
            {
                Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
                Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),
            }
        )

        self.settings.set_optical_settings(
            {
                Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
                Tags.OPTICAL_MODEL_BINARY_PATH: self.path_manager.get_mcx_binary_path(),
                Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
                Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
            }
        )

        self.pipeline = [
            sp.SegmentationBasedAdapter(self.settings),
            sp.MCXAdapter(self.settings),
        ]

    def perform_test(self):
        sp.simulate(
            self.pipeline,
            self.settings,
            sp.RSOMExplorerP50(
                element_spacing_mm=2.0,
                number_elements_y=10,
                number_elements_x=20,
                device_position_mm=np.asarray([20, 10, 0]),
            ),
        )

    def tear_down(self):
        os.remove(self.settings[Tags.SIMPA_OUTPUT_FILE_PATH])

    def visualise_result(self, show_figure_on_screen=True, save_path=None):

        if show_figure_on_screen:
            save_path = None
        else:
            save_path = save_path + f"{self.volume_name}.png"

        sp.visualise_data(
            path_to_hdf5_file=self.path_manager.get_hdf5_file_save_path()
            + "/"
            + self.volume_name
            + ".hdf5",
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
            log_scale=False,
        )


def create_segmentation_mask(volume_x_mm, volume_y_mm, volume_z_mm, spacing_mm):
    """
    Creates a simple segmentation mask with a muscular background, an epidermis layer,
    and two blood vessels at different depths.
    """
    nx = int(round(volume_x_mm / spacing_mm))
    ny = int(round(volume_y_mm / spacing_mm))
    nz = int(round(volume_z_mm / spacing_mm))

    # background (muscle): class 0
    seg = np.zeros((nx, ny, nz), dtype=int)

    # epidermis: class 1, 1mm at the top of the volume
    epidermis_end_pix = max(1, int(round(1.0 / spacing_mm)))
    seg[:, :, :epidermis_end_pix] = 1

    # shallow blood vessel: class 2, cylinder at z=5mm, radius 2mm
    vessel1_x_center = volume_x_mm / 2 - 8
    vessel1_z_center = 5.0
    vessel1_radius = 2.0

    # deep blood vessel: class 3, cylinder at z=12mm, radius 3mm
    vessel2_x_center = volume_x_mm / 2 + 5
    vessel2_z_center = 12.0
    vessel2_radius = 3.0

    x_coords = (np.arange(nx) + 0.5) * spacing_mm
    z_coords = (np.arange(nz) + 0.5) * spacing_mm
    xx, zz = np.meshgrid(x_coords, z_coords, indexing="ij")

    dist1 = np.sqrt((xx - vessel1_x_center) ** 2 + (zz - vessel1_z_center) ** 2)
    dist2 = np.sqrt((xx - vessel2_x_center) ** 2 + (zz - vessel2_z_center) ** 2)

    vessel1_mask_2d = dist1 <= vessel1_radius
    vessel2_mask_2d = dist2 <= vessel2_radius

    # repeat accross the y-dimension
    for iy in range(ny):
        seg[:, iy, :][vessel1_mask_2d] = 2
        seg[:, iy, :][vessel2_mask_2d] = 3

    return seg


class SegmentationLoaderMSOTAcuityEchoTest(ManualIntegrationTestClass):
    """
    Tests the MSOTAcuityEcho device with the segmentation-based volume creator,
    including the update_settings_for_use_of_segmentation_based_volume_creator method
    which adds mediprene, heavy water, and optionally US gel layers.
    PV_EFFECTS enables partial-volume effects in the SegmentationBasedAdapter
    FRACTION_MAP can be used to simulate a pre-computed 4D fraction map, where partial volume effects can already be included
    """

    PV_EFFECTS = False
    FRACTION_MAP = False

    def setup(self):
        self.path_manager = sp.PathManager()

        self.volume_name = f"SegmentationMSOTAcuityEchoTest{'_PV' if self.PV_EFFECTS else ''}{'_4D' if self.FRACTION_MAP else ''}"

        target_spacing = 0.5
        volume_x_mm = 40.0
        volume_y_mm = 20.0
        volume_z_mm = 25.0

        np.random.seed(4711)

        segmentation_volume_mask = create_segmentation_mask(
            volume_x_mm, volume_y_mm, volume_z_mm, target_spacing
        )

        if self.FRACTION_MAP:
            # transform into 4d fraction map and simulate existing partial volume effects via smoothing
            # One-hot encode to [C, X, Y, Z]
            # Labels must be dense (0...N-1) so that channel index == label value, which is
            # required by update_settings_for_use_of_segmentation_based_volume_creator.
            num_classes = int(segmentation_volume_mask.max()) + 1
            seg_4d = np.zeros(
                (num_classes, *segmentation_volume_mask.shape), dtype=np.float32
            )
            for c in range(num_classes):
                seg_4d[c] = (segmentation_volume_mask == c).astype(np.float32)
            if self.PV_EFFECTS:
                for c in range(num_classes):
                    seg_4d[c] = uniform_filter(seg_4d[c], size=3).astype(np.float32)
            segmentation_volume_mask = seg_4d

        def segmentation_class_mapping():
            ret_dict = dict()
            ret_dict[0] = sp.TissueLibrary.muscle()
            ret_dict[1] = sp.TissueLibrary.epidermis()
            ret_dict[2] = sp.TissueLibrary.blood()
            ret_dict[3] = sp.TissueLibrary.blood()
            return ret_dict

        self.settings = sp.Settings()
        self.settings[Tags.SIMULATION_PATH] = (
            self.path_manager.get_hdf5_file_save_path()
        )
        self.settings[Tags.VOLUME_NAME] = self.volume_name
        self.settings[Tags.RANDOM_SEED] = 4711
        self.settings[Tags.WAVELENGTHS] = [700, 800]
        self.settings[Tags.SPACING_MM] = target_spacing
        self.settings[Tags.DIM_VOLUME_X_MM] = volume_x_mm
        self.settings[Tags.DIM_VOLUME_Y_MM] = volume_y_mm
        self.settings[Tags.DIM_VOLUME_Z_MM] = volume_z_mm

        self.settings.set_volume_creation_settings(
            {
                Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
                Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),
                Tags.US_GEL: True,
                Tags.CONSIDER_PARTIAL_VOLUME: self.PV_EFFECTS
                and not self.FRACTION_MAP,  # only for 3D map. 4D map already smoothed if PV_EFFECTS is True.
            }
        )

        self.settings.set_optical_settings(
            {
                Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
                Tags.OPTICAL_MODEL_BINARY_PATH: self.path_manager.get_mcx_binary_path(),
                Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
                Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
            }
        )

        # Set up the MSOTAcuityEcho device
        self.device = sp.MSOTAcuityEcho(
            device_position_mm=np.array([volume_x_mm / 2, volume_y_mm / 2, 0])
        )

        # Use update_settings_for_use_of_segmentation_based_volume_creator to add
        # mediprene, heavy water, and us gel (US gel controlled via Tags.US_GEL).
        self.device.update_settings_for_use_of_segmentation_based_volume_creator(
            self.settings,
        )

        self.pipeline = [
            sp.SegmentationBasedAdapter(self.settings),
            sp.MCXAdapter(self.settings),
        ]

    def perform_test(self):
        sp.simulate(self.pipeline, self.settings, self.device)

    def tear_down(self):
        os.remove(self.settings[Tags.SIMPA_OUTPUT_FILE_PATH])

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        if show_figure_on_screen:
            save_path = None
        else:
            save_path = save_path + f"{self.volume_name}.png"

        sp.visualise_data(
            path_to_hdf5_file=self.path_manager.get_hdf5_file_save_path()
            + "/"
            + self.volume_name
            + ".hdf5",
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
            log_scale=False,
        )


if __name__ == "__main__":
    print("Running SegmentationLoaderRSOMExplorerTest...")
    test_rsom = SegmentationLoaderRSOMExplorerTest()
    test_rsom.run_test(show_figure_on_screen=False)

    print("Running SegmentationLoaderMSOTAcuityEchoTest (3D label map)...")
    test_msot = SegmentationLoaderMSOTAcuityEchoTest()
    test_msot.run_test(show_figure_on_screen=False)

    print("Running SegmentationLoaderMSOTAcuityEchoTest (3D label map, PV effects)...")
    test_msot_pv = SegmentationLoaderMSOTAcuityEchoTest()
    test_msot_pv.PV_EFFECTS = True
    test_msot_pv.run_test(show_figure_on_screen=False)

    print(
        "Running SegmentationLoaderMSOTAcuityEchoTest (4D fraction map including PV effects)..."
    )
    test_msot_4d = SegmentationLoaderMSOTAcuityEchoTest()
    test_msot_4d.PV_EFFECTS = True
    test_msot_4d.FRACTION_MAP = True
    test_msot_4d.run_test(show_figure_on_screen=False)
