# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
from skimage.data import shepp_logan_phantom
from scipy.ndimage import zoom

import os
from argparse import ArgumentParser
from simpa.utils.profiling import profile


# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().


DEVICE_CHOICES = ["RSOMExplorerP50", "MSOTAcuityEcho"]


def create_rsomexplorer_sample_segmentation(spacing, input_spacing=0.2):
    """
    Creates a segmentation volume from the Shepp-Logan phantom.
    """
    label_mask = shepp_logan_phantom()
    label_mask = np.digitize(label_mask, bins=np.linspace(0.0, 1.0, 11), right=True)
    label_mask = label_mask[100:300, 100:300]
    label_mask = np.reshape(label_mask, (label_mask.shape[0], 1, label_mask.shape[1]))

    segmentation_volume_tiled = np.tile(label_mask, (1, 128, 1))
    segmentation_volume_mask = sp.round_x5_away_from_zero(
        zoom(segmentation_volume_tiled, input_spacing / spacing, order=0)
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
        return ret_dict

    return segmentation_volume_mask, segmentation_class_mapping()


def create_msotacuity_sample_segmentation(
    volume_x_mm, volume_y_mm, volume_z_mm, spacing_mm
):
    """
    Use the sample segmentation mask from manual tests, with a muscular background, an epidermis layer,
    and two blood vessels at different depths.
    """

    from simpa_tests.manual_tests.volume_creation.SegmentationLoader import (
        create_segmentation_mask,
    )

    seg = create_segmentation_mask(volume_x_mm, volume_y_mm, volume_z_mm, spacing_mm)

    def segmentation_class_mapping():
        ret_dict = dict()
        ret_dict[0] = sp.TissueLibrary.muscle()
        ret_dict[1] = sp.TissueLibrary.epidermis()
        ret_dict[2] = sp.TissueLibrary.blood()
        ret_dict[3] = sp.TissueLibrary.blood()
        return ret_dict

    return seg, segmentation_class_mapping()


def setup_rsom_device(settings, spacing, partial_volume=False):
    """Set up RSOMExplorerP50 device."""
    seg, mapping = create_rsomexplorer_sample_segmentation(spacing)

    settings[Tags.VOLUME_NAME] = "SegmentationRSOMExplorer"
    settings[Tags.DIM_VOLUME_X_MM] = seg.shape[0] * spacing
    settings[Tags.DIM_VOLUME_Y_MM] = seg.shape[1] * spacing
    settings[Tags.DIM_VOLUME_Z_MM] = seg.shape[2] * spacing

    volume_creation_settings = {
        Tags.INPUT_SEGMENTATION_VOLUME: seg,
        Tags.SEGMENTATION_CLASS_MAPPING: mapping,
        Tags.CONSIDER_PARTIAL_VOLUME: partial_volume,
    }

    settings.set_volume_creation_settings(volume_creation_settings)

    device = sp.RSOMExplorerP50(element_spacing_mm=1.0)
    return device


def setup_msot_device(settings, spacing, partial_volume=False):
    """Set up MSOTAcuityEcho device."""
    volume_x_mm = 40.0
    volume_y_mm = 20.0
    volume_z_mm = 25.0

    seg, mapping = create_msotacuity_sample_segmentation(
        volume_x_mm, volume_y_mm, volume_z_mm, spacing
    )

    settings[Tags.VOLUME_NAME] = "SegmentationMSOTAcuityEcho"
    settings[Tags.DIM_VOLUME_X_MM] = volume_x_mm
    settings[Tags.DIM_VOLUME_Y_MM] = volume_y_mm
    settings[Tags.DIM_VOLUME_Z_MM] = volume_z_mm

    volume_creation_settings = {
        Tags.INPUT_SEGMENTATION_VOLUME: seg,
        Tags.SEGMENTATION_CLASS_MAPPING: mapping,
        Tags.US_GEL: True,
        Tags.CONSIDER_PARTIAL_VOLUME: partial_volume,
    }

    settings.set_volume_creation_settings(volume_creation_settings)

    device = sp.MSOTAcuityEcho(
        device_position_mm=np.array([volume_x_mm / 2, volume_y_mm / 2, 0])
    )
    device.update_settings_for_use_of_segmentation_based_volume_creator(settings)
    return device


@profile
def run_segmentation_loader(
    device_name: str = "RSOMExplorerP50",
    spacing: float | int = 0.5,
    partial_volume: bool = False,
    path_manager=None,
    visualise: bool = True,
):
    """
    Runs a segmentation-based optical simulation.

    :param device_name: Device to use, one of "RSOMExplorerP50" or "MSOTAcuityEcho"
    :param spacing: The simulation spacing between voxels in mm
    :param partial_volume: If True, enables partial-volume blending at tissue boundaries
    :param path_manager: the path manager to be used, typically sp.PathManager
    :param visualise: If True, the simulation result will be plotted
    """
    if path_manager is None:
        path_manager = sp.PathManager()

    np.random.seed(4711)

    settings = sp.Settings()
    settings[Tags.SIMULATION_PATH] = path_manager.get_hdf5_file_save_path()
    settings[Tags.RANDOM_SEED] = 4711
    settings[Tags.WAVELENGTHS] = [700, 800]
    settings[Tags.SPACING_MM] = spacing

    if device_name == "MSOTAcuityEcho":
        device = setup_msot_device(settings, spacing, partial_volume)
    elif device_name == "RSOMExplorerP50":
        device = setup_rsom_device(settings, spacing, partial_volume)
    else:
        raise ValueError(f"Unknown device: {device_name}. Choose from {DEVICE_CHOICES}")

    settings.set_optical_settings(
        {
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        }
    )

    pipeline = [
        sp.SegmentationBasedAdapter(settings),
        sp.MCXAdapter(settings),
    ]

    sp.simulate(pipeline, settings, device)

    if Tags.WAVELENGTH in settings:
        WAVELENGTH = settings[Tags.WAVELENGTH]
    else:
        WAVELENGTH = 700

    if visualise:
        sp.visualise_data(
            path_to_hdf5_file=settings[Tags.SIMPA_OUTPUT_FILE_PATH],
            wavelength=WAVELENGTH,
            show_initial_pressure=True,
            show_segmentation_map=True,
            show_absorption=True,
            show_fluence=True,
            save_path=path_manager.get_hdf5_file_save_path()
            + "/"
            + settings[Tags.VOLUME_NAME]
            + ".png",
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run segmentation-based simulation with a configurable device"
    )
    parser.add_argument(
        "--device",
        default="MSOTAcuityEcho",
        choices=DEVICE_CHOICES,
        help="the device to use",
    )
    parser.add_argument(
        "--spacing", default=0.5, type=float, help="the voxel spacing in mm"
    )
    parser.add_argument(
        "--partial_volume",
        action="store_true",
        default=True,
        help="enable partial-volume blending at tissue boundaries",
    )
    parser.add_argument(
        "--path_manager",
        default=None,
        help="the path manager, None uses sp.PathManager",
    )
    parser.add_argument(
        "--visualise", default=True, type=bool, help="whether to visualise the result"
    )
    config = parser.parse_args()

    run_segmentation_loader(
        device_name=config.device,
        spacing=config.spacing,
        partial_volume=config.partial_volume,
        path_manager=config.path_manager,
        visualise=config.visualise,
    )
