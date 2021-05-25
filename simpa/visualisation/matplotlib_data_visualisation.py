"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from simpa.utils import SegmentationClasses, Tags
from simpa.utils.settings import Settings


def visualise_data(path_to_hdf5_file: str, wavelength: int,
                   show_absorption=True,
                   show_scattering=False,
                   show_anisotropy=False,
                   show_speed_of_sound=False,
                   show_tissue_density=False,
                   show_fluence=False,
                   show_initial_pressure=True,
                   show_time_series_data=True,
                   show_reconstructed_data=True,
                   show_segmentation_map=True,
                   log_scale=True,
                   show_xz_only=False):

    file = load_hdf5(path_to_hdf5_file)

    fluence = None
    initial_pressure = None
    time_series_data = None
    reconstructed_data = None

    simulation_result_data = file['simulations']
    simulation_properties = simulation_result_data['simulation_properties']
    absorption = simulation_properties['mua'][str(wavelength)]
    scattering = simulation_properties['mus'][str(wavelength)]
    anisotropy = simulation_properties['g'][str(wavelength)]
    segmentation_map = simulation_properties['seg']
    speed_of_sound = simulation_properties['sos']
    density = simulation_properties['density']

    if "optical_forward_model_output" in simulation_result_data:
        optical_data = simulation_result_data['optical_forward_model_output']
        if "fluence" in optical_data and "initial_pressure" in optical_data:
            fluence = optical_data['fluence'][str(wavelength)]
            initial_pressure = optical_data['initial_pressure'][str(wavelength)]

    if "time_series_data" in simulation_result_data:
        time_series_data = simulation_result_data["time_series_data"][str(wavelength)]

    if "time_series_data" in simulation_result_data:
        reconstructed_data = simulation_result_data["reconstructed_data"][str(wavelength)]["reconstructed_data"]

    cmap_label_names, cmap_label_values, cmap = get_segmentation_colormap()

    data_to_show = []
    data_item_names = []
    cmaps = []
    logscales = []

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

    plt.figure()
    for i in range(len(data_to_show)):

        plt.subplot(num_rows, len(data_to_show), i+1)
        plt.title(data_item_names[i])
        if len(np.shape(data_to_show[i])) > 2:
            pos = int(np.shape(data_to_show[i])[1] / 2)
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

