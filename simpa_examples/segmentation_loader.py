# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
from skimage.data import shepp_logan_phantom
from scipy.ndimage import zoom

# FIXME temporary workaround for newest Intel architectures
import os
from argparse import ArgumentParser
from simpa.utils.profiling import profile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().


@profile
def run_segmentation_loader(spacing: float | int = 1.0, input_spacing: float | int = 0.2, path_manager=None,
                            visualise: bool = True):
    """

    :param spacing: The simulation spacing between voxels in mm
    :param input_spacing: The input spacing between voxels in mm
    :param path_manager: the path manager to be used, typically sp.PathManager
    :param visualise: If VISUALIZE is set to True, the reconstruction result will be plotted
    :return: a run through of the example
    """
    if path_manager is None:
        path_manager = sp.PathManager()

    label_mask = shepp_logan_phantom()

    label_mask = np.digitize(label_mask, bins=np.linspace(0.0, 1.0, 11), right=True)

    label_mask = np.reshape(label_mask, (400, 1, 400))

    segmentation_volume_tiled = np.tile(label_mask, (1, 128, 1))
    segmentation_volume_mask = np.round(zoom(segmentation_volume_tiled, input_spacing/spacing,
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
        return ret_dict

    settings = sp.Settings()
    settings[Tags.SIMULATION_PATH] = path_manager.get_hdf5_file_save_path()
    settings[Tags.VOLUME_NAME] = "SegmentationTest"
    settings[Tags.RANDOM_SEED] = 1234
    settings[Tags.WAVELENGTHS] = [700]
    settings[Tags.SPACING_MM] = spacing
    settings[Tags.DIM_VOLUME_X_MM] = 400 / (spacing / input_spacing)
    settings[Tags.DIM_VOLUME_Y_MM] = 128 / (spacing / input_spacing)
    settings[Tags.DIM_VOLUME_Z_MM] = 400 / (spacing / input_spacing)

    settings.set_volume_creation_settings({
        Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
        Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),

    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    })

    pipeline = [
        sp.SegmentationBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings)
    ]

    sp.simulate(pipeline, settings, sp.RSOMExplorerP50(element_spacing_mm=1.0))

    if Tags.WAVELENGTH in settings:
        WAVELENGTH = settings[Tags.WAVELENGTH]
    else:
        WAVELENGTH = 700

    if visualise:
        sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + "SegmentationTest" + ".hdf5",
                          wavelength=WAVELENGTH,
                          show_initial_pressure=True,
                          show_segmentation_map=True)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run the segmentation loader example')
    parser.add_argument("--spacing", default=1, type=float, help='the voxel spacing in mm')
    parser.add_argument("--input_spacing", default=0.2, type=float, help='the input spacing in mm')
    parser.add_argument("--path_manager", default=None, help='the path manager, None uses sp.PathManager')
    parser.add_argument("--visualise", default=True, type=bool, help='whether to visualise the result')
    config = parser.parse_args()

    run_segmentation_loader(spacing=config.spacing, input_spacing=config.input_spacing,
                            path_manager=config.path_manager, visualise=config.visualise)
