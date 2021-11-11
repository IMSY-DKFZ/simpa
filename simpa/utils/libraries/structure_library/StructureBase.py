# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod

import numpy as np

from simpa.log import Logger
from simpa.utils import Settings, Tags, get_functional_from_deformation_settings
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.tissue_properties import TissueProperties


class GeometricalStructure:
    """
    Base class for all model-based structures for ModelBasedVolumeCreator. A GeometricalStructure has an internal
    representation of its own geometry. This is represented by self.geometrical_volume which is a 3D array that defines
    for every voxel within the simulation volume if it is enclosed in the GeometricalStructure or if it is outside.
    Most of the GeometricalStructures implement a partial volume effect. So if a voxel has the value 1, it is completely
    enclosed by the GeometricalStructure. If a voxel has a value between 0 and 1, that fraction of the volume is
    occupied by the GeometricalStructure. If a voxel has the value 0, it is outside of the GeometricalStructure.
    """

    def __init__(self, global_settings: Settings,
                 single_structure_settings: Settings = None):

        self.logger = Logger()

        self.voxel_spacing = global_settings[Tags.SPACING_MM]
        volume_x_dim = int(np.round(global_settings[Tags.DIM_VOLUME_X_MM] / self.voxel_spacing))
        volume_y_dim = int(np.round(global_settings[Tags.DIM_VOLUME_Y_MM] / self.voxel_spacing))
        volume_z_dim = int(np.round(global_settings[Tags.DIM_VOLUME_Z_MM] / self.voxel_spacing))
        self.volume_dimensions_voxels = np.asarray([volume_x_dim, volume_y_dim, volume_z_dim])

        self.volume_dimensions_mm = self.volume_dimensions_voxels * self.voxel_spacing
        self.do_deformation = (Tags.SIMULATE_DEFORMED_LAYERS in global_settings.get_volume_creation_settings() and
                               global_settings.get_volume_creation_settings()[Tags.SIMULATE_DEFORMED_LAYERS])

        if (Tags.ADHERE_TO_DEFORMATION in single_structure_settings and
                not single_structure_settings[Tags.ADHERE_TO_DEFORMATION]):
            self.do_deformation = False

        self.logger.debug(f"This structure will simulate deformations: {self.do_deformation}")

        if self.do_deformation and Tags.DEFORMED_LAYERS_SETTINGS in global_settings.get_volume_creation_settings():
            self.deformation_functional_mm = get_functional_from_deformation_settings(
                global_settings.get_volume_creation_settings()[Tags.DEFORMED_LAYERS_SETTINGS])
        else:
            self.deformation_functional_mm = None

        self.logger.debug(f"This structure's deformation functional: {self.deformation_functional_mm}")

        if single_structure_settings is None:
            self.molecule_composition = MolecularComposition()
            self.priority = 0
            return

        if Tags.PRIORITY in single_structure_settings:
            self.priority = single_structure_settings[Tags.PRIORITY]

        self.partial_volume = single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME]

        self.molecule_composition = single_structure_settings[Tags.MOLECULE_COMPOSITION]
        self.molecule_composition.update_internal_properties()

        self.geometrical_volume = np.zeros(self.volume_dimensions_voxels)
        self.params = self.get_params_from_settings(single_structure_settings)
        self.fill_internal_volume()

    def fill_internal_volume(self):
        """
        Fills self.geometrical_volume of the GeometricalStructure.
        """
        indices, values = self.get_enclosed_indices()
        self.geometrical_volume[indices] = values

    @abstractmethod
    def get_enclosed_indices(self):
        """
        Gets indices of the voxels that are either entirely or partially occupied by the GeometricalStructure.
        :return: mask for a numpy array
        """
        pass

    @abstractmethod
    def get_params_from_settings(self, single_structure_settings):
        """
        Gets all the parameters required for the specific GeometricalStructure.
        :param single_structure_settings: Settings which describe the specific GeometricalStructure.
        :return: Tuple of parameters
        """
        pass

    def properties_for_wavelength(self, wavelength) -> TissueProperties:
        """
        Returns the values corresponding to each optical/acoustic property used in SIMPA.
        :param wavelength: Wavelength of the queried properties
        :return: optical/acoustic properties
        """
        return self.molecule_composition.get_properties_for_wavelength(wavelength)

    @abstractmethod
    def to_settings(self) -> Settings:
        """
        Creates a Settings dictionary which contains all the parameters needed to create the same GeometricalStructure
        again.
        :return : A tuple containing the settings key and the needed entries
        """
        settings_dict = Settings()
        settings_dict[Tags.PRIORITY] = self.priority
        settings_dict[Tags.STRUCTURE_TYPE] = self.__class__.__name__
        settings_dict[Tags.CONSIDER_PARTIAL_VOLUME] = self.partial_volume
        settings_dict[Tags.MOLECULE_COMPOSITION] = self.molecule_composition
        return settings_dict
