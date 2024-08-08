# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib as mpl
import numpy as np
from simpa.utils import SegmentationClasses, Tags
from simpa.utils.path_manager import PathManager
from simpa.utils.settings import Settings
from simpa.utils import get_data_field_from_simpa_output
from simpa.log import Logger
from abc import ABC


class VisualiseData:
    """
    A class to visualize data stored in an HDF5 file. It provides options to display various
    physical properties such as absorption, scattering, anisotropy, speed of sound, tissue density,
    fluence, initial pressure, time series data, reconstructed data, segmentation maps,
    oxygenation, blood volume fraction, and linear unmixing results.

    The class supports visualizing data in 2D (xz-plane and zy-plane), and includes features such as
    logarithmic scaling, saving visualizations, and adding a slider for wavelength selection.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure object for plotting.
    axes : numpy.ndarray
        Array of axes for subplots.
    logger : Logger
        Logger object for handling logging.

    Methods
    -------
    plot_data_for_wavelength(wavelength, colour_bar=True):
        Plots data for a specific wavelength.

    get_segmentation_colormap():
        Returns the colormap for the segmentation map.
    """

    def __init__(self,
                 wavelengths: list = None,
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
                 show_blood_volume_fraction=False,
                 show_linear_unmixing_sO2=False,
                 show_diffuse_reflectance=False,
                 log_scale=False,
                 show_xz_only=False,
                 save_path=None,
                 add_wavelengths_slider: bool = True):
        """
        :param wavelengths: A list of wavelengths to be visualized.
        :param path_to_hdf5_file: Path to the HDF5 file containing the data. If not provided, it is constructed from settings and path_manager.
        :param settings: An instance of Settings containing configuration parameters, including the volume name and wavelengths.
        :param path_manager: An instance of PathManager used to manage file paths. It is used to generate the path to the HDF5 file if not provided.
        :param show_absorption: If True, display the absorption coefficient.
        :param show_scattering: If True, display the scattering coefficient.
        :param show_anisotropy: If True, display the anisotropy.
        :param show_speed_of_sound: If True, display the speed of sound.
        :param show_tissue_density: If True, display the tissue density.
        :param show_fluence: If True, display the fluence.
        :param show_initial_pressure: If True, display the initial pressure.
        :param show_time_series_data: If True, display the time series data.
        :param show_reconstructed_data: If True, display the reconstructed data.
        :param show_segmentation_map: If True, display the segmentation map.
        :param show_oxygenation: If True, display the oxygenation.
        :param show_blood_volume_fraction: If True, display the blood volume fraction.
        :param show_linear_unmixing_sO2: If True, display the linear unmixing of oxygen saturation (sO2).
        :param show_diffuse_reflectance: If True, display the diffuse reflectance.
        :param log_scale: If True, use a logarithmic scale for the data.
        :param show_xz_only: If True, display only the xz-plane.
        :param save_path: Path to save the visualization. If None, the visualization will be displayed instead of saved.
        :param add_wavelengths_slider: If True, add a slider for wavelength selection to the visualization.

        :raises ValueError: If both path_to_hdf5_file and the combination of settings and path_manager are None.
        """

        super().__init__()

        if path_to_hdf5_file is None and (settings is None or path_manager is None):
            raise ValueError("Either the path_to_hdf5_file or the given settings and path_manager must not be None!")

        if path_to_hdf5_file is None:
            path_to_hdf5_file = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + ".hdf5"

        if settings is not None and Tags.WAVELENGTHS in settings:
            wavelengths = settings[Tags.WAVELENGTHS]

        self.show_absorption = show_absorption
        self.show_scattering = show_scattering
        self.show_anisotropy = show_anisotropy
        self.show_speed_of_sound = show_speed_of_sound
        self.show_tissue_density = show_tissue_density
        self.show_fluence = show_fluence
        self.show_initial_pressure = show_initial_pressure
        self.show_time_series_data = show_time_series_data
        self.show_reconstructed_data = show_reconstructed_data
        self.show_segmentation_map = show_segmentation_map
        self.show_oxygenation = show_oxygenation
        self.show_blood_volume_fraction = show_blood_volume_fraction
        self.show_linear_unmixing_sO2 = show_linear_unmixing_sO2
        self.show_diffuse_reflectance = show_diffuse_reflectance

        num_cols = 0
        for attribute in dir(self):
            if not attribute.startswith('_') and getattr(self, attribute) is True:
                num_cols += 1

        self.show_xz_only = show_xz_only
        self.path_to_hdf5_file = load_hdf5(path_to_hdf5_file)
        self.log_scale = log_scale
        self.save_path = save_path
        self.logger = Logger()

        if self.show_xz_only:
            num_rows = 1
        else:
            num_rows = 2
        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*3.5))

        self.plot_data_for_wavelength(wavelengths[0])
        plt.tight_layout()

        if len(wavelengths) != 1 and add_wavelengths_slider:
            # find the step in the wavelengths
            wavelength_step = wavelengths[1] - wavelengths[0]

            # Make a slider to control the wavelength
            self.fig.subplots_adjust(left=0.25)
            ax_wavelength = self.fig.add_axes([0.1, 0.25, 0.0225, 0.63])
            wavelength_slider = Slider(
                ax=ax_wavelength,
                label="Wavelength (nm)",
                valmin=np.min(wavelengths),
                valmax=np.max(wavelengths),
                valinit=np.min(wavelengths),
                valstep=wavelength_step,
                orientation="vertical"
            )

            # Function to update the value with the slider
            def update(val):
                self.plot_data_for_wavelength(val, colour_bar=False)
                self.fig.canvas.draw_idle()

            # update on changed slider value
            wavelength_slider.on_changed(update)

            plt.show()

        if save_path is not None:
            plt.savefig(save_path, dpi=500)
        else:
            plt.show()
        plt.close()

    def plot_data_for_wavelength(self, wavelength: int = None, colour_bar: bool = True):
        """
        Plots the data for a specific wavelength. The method handles various types of data,
        including absorption, scattering, fluence, initial pressure, etc.

        :param wavelength: The wavelength for which data should be plotted.
        :type wavelength: int
        :param colour_bar: Whether to display the color bar.
        :type colour_bar: bool
        """

        fluence = None
        initial_pressure = None
        time_series_data = None
        reconstructed_data = None
        oxygenation = None
        blood_volume_fraction = None
        linear_unmixing_sO2 = None
        diffuse_reflectance = None
        diffuse_reflectance_position = None

        absorption = get_data_field_from_simpa_output(
            self.path_to_hdf5_file, Tags.DATA_FIELD_ABSORPTION_PER_CM, wavelength)
        scattering = get_data_field_from_simpa_output(
            self.path_to_hdf5_file, Tags.DATA_FIELD_SCATTERING_PER_CM, wavelength)
        anisotropy = get_data_field_from_simpa_output(self.path_to_hdf5_file, Tags.DATA_FIELD_ANISOTROPY, wavelength)
        segmentation_map = get_data_field_from_simpa_output(self.path_to_hdf5_file, Tags.DATA_FIELD_SEGMENTATION)
        speed_of_sound = get_data_field_from_simpa_output(self.path_to_hdf5_file, Tags.DATA_FIELD_SPEED_OF_SOUND)
        density = get_data_field_from_simpa_output(self.path_to_hdf5_file, Tags.DATA_FIELD_DENSITY)

        if self.show_fluence:
            try:
                fluence = get_data_field_from_simpa_output(self.path_to_hdf5_file, Tags.DATA_FIELD_FLUENCE, wavelength)
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.DATA_FIELD_FLUENCE) + " was not in the simpa output.")
                show_fluence = False
                fluence = None

        if self.show_diffuse_reflectance:
            try:
                diffuse_reflectance = get_data_field_from_simpa_output(self.path_to_hdf5_file,
                                                                       Tags.DATA_FIELD_DIFFUSE_REFLECTANCE,
                                                                       wavelength)
                diffuse_reflectance_position = get_data_field_from_simpa_output(self.path_to_hdf5_file,
                                                                                Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS,
                                                                                wavelength)
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.DATA_FIELD_FLUENCE) + " was not in the simpa output.")
                show_fluence = False
                fluence = None

        if self.show_initial_pressure:
            try:
                initial_pressure = get_data_field_from_simpa_output(
                    self.path_to_hdf5_file, Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength)
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.DATA_FIELD_INITIAL_PRESSURE) +
                                     " was not in the simpa output.")
                show_initial_pressure = False
                initial_pressure = None

        if self.show_time_series_data:
            try:
                time_series_data = get_data_field_from_simpa_output(
                    self.path_to_hdf5_file, Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.DATA_FIELD_TIME_SERIES_DATA) +
                                     " was not in the simpa output.")
                show_time_series_data = False
                time_series_data = None

        if self.show_reconstructed_data:
            try:
                reconstructed_data = get_data_field_from_simpa_output(
                    self.path_to_hdf5_file, Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength)
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.DATA_FIELD_RECONSTRUCTED_DATA) +
                                     " was not in the simpa output.")
                show_reconstructed_data = False
                reconstructed_data = None

        if self.show_oxygenation:
            try:
                oxygenation = get_data_field_from_simpa_output(
                    self.path_to_hdf5_file, Tags.DATA_FIELD_OXYGENATION, wavelength)
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.DATA_FIELD_OXYGENATION) + " was not in the simpa output.")
                show_oxygenation = False
                oxygenation = None

        if self.show_blood_volume_fraction:
            try:
                blood_volume_fraction = get_data_field_from_simpa_output(
                    self.path_to_hdf5_file, Tags.DATA_FIELD_BLOOD_VOLUME_FRACTION, wavelength)
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.DATA_FIELD_BLOOD_VOLUME_FRACTION) +
                                     " was not in the simpa output.")
                show_blood_volume_fraction = False
                blood_volume_fraction = None

        if self.show_linear_unmixing_sO2:
            try:
                linear_unmixing_output = get_data_field_from_simpa_output(
                    self.path_to_hdf5_file, Tags.LINEAR_UNMIXING_RESULT)
                linear_unmixing_sO2 = linear_unmixing_output["sO2"]
            except KeyError as e:
                self.logger.critical("The key " + str(Tags.LINEAR_UNMIXING_RESULT) + " was not in the simpa output or blood "
                                     "oxygen saturation was not computed.")
                show_linear_unmixing_sO2 = False
                linear_unmixing_sO2 = None

        cmap_label_names, cmap_label_values, cmap = self.get_segmentation_colourmap()

        data_to_show = []
        data_item_names = []
        cmaps = []
        logscales = []

        if diffuse_reflectance is not None and self.show_diffuse_reflectance:
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

        if absorption is not None and self.show_absorption:
            data_to_show.append(absorption)
            data_item_names.append("Absorption Coefficient")
            cmaps.append("gray")
            logscales.append(True and self.log_scale)
        if scattering is not None and self.show_scattering:
            data_to_show.append(scattering)
            data_item_names.append("Scattering Coefficient")
            cmaps.append("gray")
            logscales.append(True and self.log_scale)
        if anisotropy is not None and self.show_anisotropy:
            data_to_show.append(anisotropy)
            data_item_names.append("Anisotropy")
            cmaps.append("gray")
            logscales.append(True and self.log_scale)
        if speed_of_sound is not None and self.show_speed_of_sound:
            data_to_show.append(speed_of_sound)
            data_item_names.append("Speed of Sound")
            cmaps.append("gray")
            logscales.append(True and self.log_scale)
        if density is not None and self.show_tissue_density:
            data_to_show.append(density)
            data_item_names.append("Density")
            cmaps.append("gray")
            logscales.append(True and self.log_scale)
        if fluence is not None and self.show_fluence:
            data_to_show.append(fluence)
            data_item_names.append("Fluence")
            cmaps.append("viridis")
            logscales.append(True and self.log_scale)
        if initial_pressure is not None and self.show_initial_pressure:
            data_to_show.append(initial_pressure)
            data_item_names.append("Initial Pressure")
            cmaps.append("viridis")
            logscales.append(True and self.log_scale)
        if time_series_data is not None and self.show_time_series_data:
            data_to_show.append(time_series_data)
            data_item_names.append("Time Series Data")
            cmaps.append("gray")
            logscales.append(False and self.log_scale)
        if reconstructed_data is not None and self.show_reconstructed_data:
            data_to_show.append(reconstructed_data)
            data_item_names.append("Reconstruction")
            cmaps.append("viridis")
            logscales.append(True and self.log_scale)
        if oxygenation is not None and self.show_oxygenation:
            data_to_show.append(oxygenation)
            data_item_names.append("Oxygenation")
            cmaps.append("viridis")
            logscales.append(False and self.log_scale)
        if blood_volume_fraction is not None and self.show_blood_volume_fraction:
            data_to_show.append(blood_volume_fraction)
            data_item_names.append("Blood Volume Fraction")
            cmaps.append("viridis")
            logscales.append(False and self.log_scale)
        if linear_unmixing_sO2 is not None and self.show_linear_unmixing_sO2:
            data_to_show.append(linear_unmixing_sO2)
            data_item_names.append("Linear Unmixed Oxygenation")
            cmaps.append("viridis")
            logscales.append(False and self.log_scale)
        if segmentation_map is not None and self.show_segmentation_map:
            data_to_show.append(segmentation_map)
            data_item_names.append("Segmentation Map")
            cmaps.append(cmap)
            logscales.append(False)

        for i in range(len(data_to_show)):

            if self.show_xz_only:
                self.axes[i].clear()
                self.axes[i].set_title(data_item_names[i])
                if len(np.shape(data_to_show[i])) > 2:
                    pos = int(np.shape(data_to_show[i])[1] / 2) - 1
                    data = data_to_show[i][:, pos, :].T
                    img = self.axes[i].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                else:
                    data = data_to_show[i][:, :].T
                    img = self.axes[i].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                if colour_bar:
                    plt.colorbar(img, ax=self.axes[i])

            elif len(data_to_show) == 1 and not self.show_xz_only:
                self.axes[0].clear()
                self.axes[0].set_title(data_item_names)
                if len(np.shape(data_to_show[i])) > 2:
                    pos = int(np.shape(data_to_show[i])[1] / 2) - 1
                    data = data_to_show[i][:, pos, :].T
                    img = self.axes[0].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                else:
                    data = data_to_show[i][:, :].T
                    img = self.axes[0].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                if colour_bar:
                    plt.colorbar(img, ax=self.axes[0])

                self.axes[1].clear()
                self.axes[1].set_title(data_item_names)
                if len(np.shape(data_to_show[i])) > 2:
                    pos = int(np.shape(data_to_show[i])[0] / 2)
                    data = data_to_show[i][pos, :, :].T
                    img = self.axes[1].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                else:
                    data = data_to_show[i][:, :].T
                    img = self.axes[1].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                if colour_bar:
                    plt.colorbar(img, ax=self.axes[1])

            elif not self.show_xz_only:
                self.axes[0][i].clear()
                self.axes[0][i].set_title(data_item_names[i])
                if len(np.shape(data_to_show[i])) > 2:
                    pos = int(np.shape(data_to_show[i])[1] / 2) - 1
                    data = data_to_show[i][:, pos, :].T
                    img = self.axes[0, i].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                else:
                    data = data_to_show[i][:, :].T
                    img = self.axes[0, i].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                if colour_bar:
                    plt.colorbar(img, ax=self.axes[0][i])

                self.axes[1][i].clear()
                self.axes[1][i].set_title(data_item_names[i])
                if len(np.shape(data_to_show[i])) > 2:
                    pos = int(np.shape(data_to_show[i])[0] / 2)
                    data = data_to_show[i][pos, :, :].T
                    img = self.axes[1][i].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                else:
                    data = data_to_show[i][:, :].T
                    img = self.axes[1][i].imshow(np.log10(data) if logscales[i] else data, cmap=cmaps[i])
                if colour_bar:
                    plt.colorbar(img, ax=self.axes[1][i])

    def get_segmentation_colourmap(self):
        """
        Generates a colormap for segmentation classes.

        This method creates a custom colormap based on the segmentation classes defined in the
        `SegmentationClasses` class. Each class is assigned a random color.

        :return: A tuple containing the names of the segmentation classes, their corresponding values, and the colormap.
        :rtype: tuple (np.ndarray, np.ndarray, mpl.colors.LinearSegmentedColormap)
            - names: An array of the segmentation class names.
            - values: An array of the segmentation class values.
            - cmap: A matplotlib colormap object created from random colors.
        """

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
