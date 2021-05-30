# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from simpa.utils import SegmentationClasses, Tags
from simpa.utils.settings import Settings
from simpa.utils import get_data_field_from_simpa_output
from simpa.log import Logger


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

    logger = Logger()
    file = load_hdf5(path_to_hdf5_file)

    fluence = None
    initial_pressure = None
    time_series_data = None
    reconstructed_data = None

    absorption = get_data_field_from_simpa_output(file, Tags.PROPERTY_ABSORPTION_PER_CM, wavelength)
    scattering = get_data_field_from_simpa_output(file, Tags.PROPERTY_SCATTERING_PER_CM, wavelength)
    anisotropy = get_data_field_from_simpa_output(file, Tags.PROPERTY_ANISOTROPY, wavelength)
    segmentation_map = get_data_field_from_simpa_output(file, Tags.PROPERTY_SEGMENTATION)
    speed_of_sound = get_data_field_from_simpa_output(file, Tags.PROPERTY_SPEED_OF_SOUND)
    density = get_data_field_from_simpa_output(file, Tags.PROPERTY_DENSITY)

    if show_fluence:
        try:
            fluence = get_data_field_from_simpa_output(file, Tags.OPTICAL_MODEL_FLUENCE, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.OPTICAL_MODEL_FLUENCE) + " was not in the simpa output.")
            raise e

    if show_initial_pressure:
        try:
            initial_pressure = get_data_field_from_simpa_output(file, Tags.OPTICAL_MODEL_INITIAL_PRESSURE, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.OPTICAL_MODEL_INITIAL_PRESSURE) + " was not in the simpa output.")
            raise e

    if show_time_series_data:
        try:
            time_series_data = get_data_field_from_simpa_output(file, Tags.TIME_SERIES_DATA, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.TIME_SERIES_DATA) + " was not in the simpa output.")
            raise e

    if show_reconstructed_data:
        try:
            reconstructed_data = get_data_field_from_simpa_output(file, Tags.RECONSTRUCTED_DATA, wavelength)
        except KeyError as e:
            logger.critical("The key " + str(Tags.RECONSTRUCTED_DATA) + " was not in the simpa output.")
            raise e

    shape = np.shape(absorption)
    x_pos = int(shape[0] / 2)
    y_pos = int(shape[1] / 2)
    z_pos = int(shape[2] / 2)

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
            data = np.rot90(data_to_show[i][:, y_pos, :], -1)
            plt.imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
        else:
            data = np.rot90(data_to_show[i][:, :], -1)
            plt.imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
        plt.colorbar()

        if not show_xz_only:
            plt.subplot(num_rows, len(data_to_show), i + 1 + len(data_to_show))
            plt.title(data_item_names[i])
            if len(np.shape(data_to_show[i])) > 2:
                data = np.rot90(data_to_show[i][x_pos, :, :], -1)
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

