# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
from skimage.data import shepp_logan_phantom
from scipy.ndimage import zoom
from skimage.transform import resize
import nrrd
import random

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

    def us_heterogeneity():
        """

        """
        scans = [2, 3, 14, 19, 32, 39, 50, 51]
        scan = random.choice(scans)

        blood_volume_fraction = sp.ImageHeterogeneity(xdim=400, ydim=200, zdim=400, spacing_mm=spacing, target_min=0,
                                                      target_max=0.05, ultrasound_image_type=Tags.MEAT_ULTRASOUND_FULL)
        blood_volume_fraction.exponential(6)
        blood_volume_fraction.invert_image()

        segmentation_mask, header = nrrd.read("./beef_ultrasound_database/segmentations/Scan_"+str(scan)+"_labels.nrrd")
        segmentation_mask = np.repeat(np.swapaxes(segmentation_mask, 1, 2), 200, axis=1).astype(np.int32)
        labels = header['org.mitk.multilabel.segmentation.labelgroups']

        # Here, we extract the labels from the nrrd file created by the MITK Workbench used to segment our data.
        lab_no = 1
        label_dict = {}
        while True:
            label, number, labels = labels.rpartition('"value":'+str(lab_no)+',')
            actual_label = label.rpartition('"name":"', )[2].rpartition('","opacity":')[0]
            if not actual_label:
                break
            label_dict[actual_label] = lab_no
            lab_no += 1

        # A fix for the fact only one has a fat layer on top.
        try:
            fat = label_dict['5 Fat']
        except KeyError:
            fat = 7

        background_dictionary = sp.Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        tissue_dict = dict()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict[label_dict['1 Heavy Water']] = sp.TISSUE_LIBRARY.heavy_water()
        tissue_dict[label_dict['2 Mediprene']] = sp.TISSUE_LIBRARY.mediprene()
        tissue_dict[label_dict['3 US Gel']] = sp.TISSUE_LIBRARY.ultrasound_gel()
        tissue_dict[label_dict['4 Muscle']] = sp.TISSUE_LIBRARY.muscle(oxygenation=0.4,
                                                                       blood_volume_fraction=blood_volume_fraction.get_map())
        tissue_dict[fat] = sp.TISSUE_LIBRARY.subcutaneous_fat()
        return tissue_dict, segmentation_mask

    def shepp_logan_phantom():
        label_mask = shepp_logan_phantom()

        label_mask = np.digitize(label_mask, bins=np.linspace(0.0, 1.0, 11), right=True)
        label_mask = label_mask[100:300, 100:300]
        label_mask = np.reshape(label_mask, (label_mask.shape[0], 1, label_mask.shape[1]))

        segmentation_volume_tiled = np.tile(label_mask, (1, 128, 1))
        segmentation_volume_mask = np.round(zoom(segmentation_volume_tiled, input_spacing / spacing,
                                                 order=0)).astype(int)

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

        return ret_dict, segmentation_volume_mask

    ###############################################
    # TODO: uncomment which example you wish to run
    volume_settings = us_heterogeneity()
    # volume_settings = shepp_logan_phantom()
    ###############################################

    settings = sp.Settings()
    settings[Tags.SIMULATION_PATH] = path_manager.get_hdf5_file_save_path()
    settings[Tags.VOLUME_NAME] = "SegmentationTest"
    settings[Tags.RANDOM_SEED] = 1234
    settings[Tags.WAVELENGTHS] = [700]
    settings[Tags.SPACING_MM] = spacing
    settings[Tags.DIM_VOLUME_X_MM] = volume_settings[1].shape[0] * spacing
    settings[Tags.DIM_VOLUME_Y_MM] = volume_settings[1].shape[1] * spacing
    settings[Tags.DIM_VOLUME_Z_MM] = volume_settings[1].shape[2] * spacing

    settings.set_volume_creation_settings({
        Tags.INPUT_SEGMENTATION_VOLUME: volume_settings[1],
        Tags.SEGMENTATION_CLASS_MAPPING: volume_settings[0],

    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    })

    pipeline = [
        sp.SegmentationBasedAdapter(settings),
        sp.MCXAdapter(settings)
    ]

    # TODO: For the device choice, uncomment the undesired device
    device = sp.RSOMExplorerP50(element_spacing_mm=1.0)
    # device = sp.MSOTAcuityEcho(device_position_mm=np.array([settings[Tags.DIM_VOLUME_X_MM] / 2,
    #                                                         settings[Tags.DIM_VOLUME_Y_MM] / 2,
    #                                                         0]))
    # device.update_settings_for_use_of_segmentation_based_volume_creator(settings, add_layers=[Tags.ADD_HEAVY_WATER],
    #                                                                     heavy_water_tag=2,
    #                                                                     current_heavy_water_depth=100 * spacing)

    sp.simulate(pipeline, settings, device)

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
