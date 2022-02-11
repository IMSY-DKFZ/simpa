# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from simpa.utils import SegmentationClasses, Tags
from simpa.utils.path_manager import PathManager
from simpa.utils.settings import Settings
from simpa.utils import get_data_field_from_simpa_output
from simpa.log import Logger


def visualise_data(wavelength: int = None,
                   path_to_hdf5_file: str = None,
                   settings: Settings = None,
                   path_manager: PathManager = None,
                   show_absorption=False,
                   show_scattering=False,
                   show_anisotropy=False,
                   show_speed_of_sound=False,
                   show_tissue_density=False,
                   show_fluence=False,
                   show_initial_pressure=False,
                   show_time_series_data=False,
                   show_reconstructed_data=False,
                   show_segmentation_map=False,
                   show_oxygenation=False,
                   show_linear_unmixing_sO2=False,
                   show_diffuse_reflectance=False,
                   log_scale=False,
                   show_xz_only=False,
                   save_path=None):

    if settings is not None and Tags.WAVELENGTHS in settings:
        if wavelength is None or wavelength not in settings[Tags.WAVELENGTHS]:
            wavelength = settings[Tags.WAVELENGTHS][0]

    if settings is not None and Tags.WAVELENGTH in settings:
        wavelength = settings[Tags.WAVELENGTH]

    if path_to_hdf5_file is None and (settings is None or path_manager is None):
        raise ValueError("Either the path_to_hdf5_file or the given settings and path_manager must not be None!")

    if path_to_hdf5_file is None:
        path_to_hdf5_file = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + ".hdf5"

    logger = Logger()
    file = load_hdf5(path_to_hdf5_file)

    fluence = None
    initial_pressure = None
    time_series_data = None
    reconstructed_data = None
    oxygenation = None
    linear_unmixing_sO2 = None
    diffuse_reflectance = None
    diffuse_reflectance_position = None

    absorption = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_ABSORPTION_PER_CM, wavelength)
    scattering = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_SCATTERING_PER_CM, wavelength)
    anisotropy = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_ANISOTROPY, wavelength)
    segmentation_map = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_SEGMENTATION)
    speed_of_sound = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_SPEED_OF_SOUND)
    density = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_DENSITY)

    if show_fluence:
        try:
            fluence = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_FLUENCE, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.DATA_FIELD_FLUENCE) + " was not in the simpa output.")
            show_fluence = False
            fluence = None

    if show_diffuse_reflectance:
        try:
            diffuse_reflectance = get_data_field_from_simpa_output(file,
                                                                   Tags.DATA_FIELD_DIFFUSE_REFLECTANCE,
                                                                   wavelength)
            diffuse_reflectance_position = get_data_field_from_simpa_output(file,
                                                                            Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS,
                                                                            wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.DATA_FIELD_FLUENCE) + " was not in the simpa output.")
            show_fluence = False
            fluence = None

    if show_initial_pressure:
        try:
            initial_pressure = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.DATA_FIELD_INITIAL_PRESSURE) + " was not in the simpa output.")
            show_initial_pressure = False
            initial_pressure = None

    if show_time_series_data:
        try:
            time_series_data = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.DATA_FIELD_TIME_SERIES_DATA) + " was not in the simpa output.")
            show_time_series_data = False
            time_series_data = None

    if show_reconstructed_data:
        try:
            reconstructed_data = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.DATA_FIELD_RECONSTRUCTED_DATA) + " was not in the simpa output.")
            show_reconstructed_data = False
            reconstructed_data = None

    if show_oxygenation:
        try:
            oxygenation = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_OXYGENATION, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.DATA_FIELD_OXYGENATION) + " was not in the simpa output.")
            show_oxygenation = False
            oxygenation = None

    if show_linear_unmixing_sO2:
        try:
            linear_unmixing_output = get_data_field_from_simpa_output(file, Tags.LINEAR_UNMIXING_RESULT)
            linear_unmixing_sO2 = linear_unmixing_output["sO2"]
        except KeyError as e:
            logger.critical("The key " + str(Tags.LINEAR_UNMIXING_RESULT) + " was not in the simpa output or blood "
                                                                            "oxygen saturation was not computed.")
            show_linear_unmixing_sO2 = False
            linear_unmixing_sO2 = None

    cmap_label_names, cmap_label_values, cmap = get_segmentation_colormap()

    data_to_show = []
    data_item_names = []
    cmaps = []
    logscales = []

    if diffuse_reflectance is not None and show_diffuse_reflectance:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title("Diffuse reflectance")
        ax.scatter(diffuse_reflectance_position[:, 0],
                   diffuse_reflectance_position[:, 1],
                   diffuse_reflectance_position[:, 2],
                   c=diffuse_reflectance,
                   cmap='RdBu',
                   antialiased=False)
        ax.set_box_aspect((2, 1, 1))
        plt.show()

    if absorption is not None and show_absorption:
        data_to_show.append(absorption)
        data_item_names.append("Absorption Coefficient")
        cmaps.append("gray")
        logscales.append(True and log_scale)
    if scattering is not None and show_scattering:
        data_to_show.append(scattering)
        data_item_names.append("Scattering Coefficient")
        cmaps.append("gray")
        logscales.append(True and log_scale)
    if anisotropy is not None and show_anisotropy:
        data_to_show.append(anisotropy)
        data_item_names.append("Anisotropy")
        cmaps.append("gray")
        logscales.append(True and log_scale)
    if speed_of_sound is not None and show_speed_of_sound:
        data_to_show.append(speed_of_sound)
        data_item_names.append("Speed of Sound")
        cmaps.append("gray")
        logscales.append(True and log_scale)
    if density is not None and show_tissue_density:
        data_to_show.append(density)
        data_item_names.append("Density")
        cmaps.append("gray")
        logscales.append(True and log_scale)
    if fluence is not None and show_fluence:
        data_to_show.append(fluence)
        data_item_names.append("Fluence")
        cmaps.append("viridis")
        logscales.append(True and log_scale)
    if initial_pressure is not None and show_initial_pressure:
        data_to_show.append(initial_pressure)
        data_item_names.append("Initial Pressure")
        cmaps.append("viridis")
        logscales.append(True and log_scale)
    if time_series_data is not None and show_time_series_data:
        data_to_show.append(time_series_data)
        data_item_names.append("Time Series Data")
        cmaps.append("gray")
        logscales.append(False and log_scale)
    if reconstructed_data is not None and show_reconstructed_data:
        data_to_show.append(reconstructed_data)
        data_item_names.append("Reconstruction")
        cmaps.append("viridis")
        logscales.append(True and log_scale)
    if oxygenation is not None and show_oxygenation:
        data_to_show.append(oxygenation)
        data_item_names.append("Oxygenation")
        cmaps.append("viridis")
        logscales.append(False and log_scale)
    if linear_unmixing_sO2 is not None and show_linear_unmixing_sO2:
        data_to_show.append(linear_unmixing_sO2)
        data_item_names.append("Linear Unmixed Oxygenation")
        cmaps.append("viridis")
        logscales.append(False and log_scale)
    if segmentation_map is not None and show_segmentation_map:
        data_to_show.append(segmentation_map)
        data_item_names.append("Segmentation Map")
        cmaps.append(cmap)
        logscales.append(False)

    if show_xz_only:
        num_rows = 1
    else:
        num_rows = 2

    plt.figure(figsize=(len(data_to_show)*4, num_rows*3.5))
    for i in range(len(data_to_show)):

        plt.subplot(num_rows, len(data_to_show), i+1)
        plt.title(data_item_names[i])
        if len(np.shape(data_to_show[i])) > 2:
            pos = int(np.shape(data_to_show[i])[1] / 2) - 1
            data = np.rot90(data_to_show[i][:, pos, :], -1)
            plt.imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
        else:
            data = np.rot90(data_to_show[i][:, :], -1)
            plt.imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
        plt.colorbar()

        if not show_xz_only:
            plt.subplot(num_rows, len(data_to_show), i + 1 + len(data_to_show))
            plt.title(data_item_names[i])
            if len(np.shape(data_to_show[i])) > 2:
                pos = int(np.shape(data_to_show[i])[0] / 2)
                data = np.rot90(data_to_show[i][pos, :, :], -1)
                plt.imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
            else:
                data = np.rot90(data_to_show[i][:, :], -1)
                plt.imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
            plt.colorbar()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    else:
        plt.show()
    plt.close()


def get_segmentation_colormap():
    values = []
    names = []

    for string in SegmentationClasses.__dict__:
        if string[0:2] != "__":
            values.append(SegmentationClasses.__dict__[string])
            names.append(string)

    values = np.asarray(values)
    names = np.asarray(names)
    sort_indexes = np.argsort(values)
    values = values[sort_indexes]
    names = names[sort_indexes]

    colors = [list(np.random.random(3)) for _ in range(len(names))]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', colors, len(names))

    return names, values, cmap
