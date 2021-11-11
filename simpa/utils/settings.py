# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.utils.serializer import SerializableSIMPAClass
from simpa.log import Logger


class Settings(dict, SerializableSIMPAClass):
    """
    The Settings class is a dictionary that contains all relevant settings for running a simulation in the SIMPA
    toolkit. It includes an automatic sanity check for input parameters using the simpa.utils.Tags class. \n
    Usage: Settings({Tags.KEY1: value1, Tags.KEY2: value2, ...})
    """

    def __init__(self, dictionary: dict = None, verbose: bool = True):
        super(Settings, self).__init__()
        self.logger = Logger()
        self.verbose = verbose
        if dictionary is None:
            dictionary = {}
        for key, value in dictionary.items():
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            super().__setitem__(key, value)
            if self.verbose:
                self.logger.warning("The key for the Settings dictionary should be a tuple in the form of "
                                    "('{}', (data_type_1, data_type_2, ...)). "
                                    "The tuple of data types specifies all possible types, the value can have.\n"
                                    "The key '{}' has been given the value {}".format(key, key, value))
            return
        elif not isinstance(key, tuple):
            raise TypeError("The key for the Settings dictionary has to be a tuple in the form of "
                            "('{}', (data_type_1, data_type_2, ...)). "
                            "The tuple of data types specifies all possible types, the value can have.".format(key))
        if isinstance(value, key[1]):
            super().__setitem__(key[0], value)
        else:
            raise ValueError("The value {} ({}) for the key '{}' has to be an instance of: "
                             "{}".format(value, type(value), key[0], key[1]))

    def __contains__(self, item):
        if super().__contains__(item) is True:
            return True
        elif isinstance(item, str) is False and super().__contains__(item[0]) is True:
            return True
        else:
            return False

    def __getitem__(self, item):
        if super().__contains__(item) is True:
            return super().__getitem__(item)
        else:
            try:
                return super().__getitem__(item[0])
            except KeyError:
                key = item[0] if isinstance(item, tuple) else item
                raise KeyError("The key '{}' is not in the Settings dictionary".format(key)) from None

    def __delitem__(self, key):
        if super().__contains__(key) is True:
            return super().__delitem__(key)
        else:
            try:
                return super().__delitem__(key[0])
            except KeyError:
                raise KeyError("The key '{}' is not in the Settings dictionary".format(key)) from None

    def get_optical_settings(self):
        """"
        Returns the settings for the optical forward model that are saved in this settings dictionary
        """
        optical_settings = self[Tags.OPTICAL_MODEL_SETTINGS]
        if isinstance(optical_settings, Settings):
            return optical_settings
        else:
            return Settings(optical_settings)

    def set_optical_settings(self, optical_settings: dict):
        """
        Replaces the currently stored optical settings with the given dictionary

        :param optical_settings: a dictionary containing the optical settings
        """
        self[Tags.OPTICAL_MODEL_SETTINGS] = Settings(optical_settings)

    def get_volume_creation_settings(self):
        """"
        Returns the settings for the optical forward model that are saved in this settings dictionary
        """
        volume_creation_settings = self[Tags.VOLUME_CREATION_MODEL_SETTINGS]
        if isinstance(volume_creation_settings, Settings):
            return volume_creation_settings
        else:
            return Settings(volume_creation_settings)

    def set_volume_creation_settings(self, volume_settings: dict):
        """
        Replaces the currently stored volume creation settings with the given dictionary

        :param volume_settings: a dictionary containing the volume creator settings
        """
        self[Tags.VOLUME_CREATION_MODEL_SETTINGS] = Settings(volume_settings)

    def get_acoustic_settings(self):
        """"
        Returns the settings for the acoustic forward model that are saved in this settings dictionary
        """
        acoustic_settings = self[Tags.ACOUSTIC_MODEL_SETTINGS]
        if isinstance(acoustic_settings, Settings):
            return acoustic_settings
        else:
            return Settings(acoustic_settings)

    def set_acoustic_settings(self, acoustic_settings: dict):
        """
        Replaces the currently stored acoustic forward model settings with the given dictionary

        :param acoustic_settings: a dictionary containing the acoustic model settings
        """
        self[Tags.ACOUSTIC_MODEL_SETTINGS] = Settings(acoustic_settings)

    def get_reconstruction_settings(self):
        """"
        Returns the settings for the reconstruction model that are saved in this settings dictionary
        """
        reconstruction_settings = self[Tags.RECONSTRUCTION_MODEL_SETTINGS]
        if isinstance(reconstruction_settings, Settings):
            return reconstruction_settings
        else:
            return Settings(reconstruction_settings)

    def set_reconstruction_settings(self, reconstruction_settings: dict):
        """
        Replaces the currently stored reconstruction model settings with the given dictionary

        :param reconstruction_settings: a dictionary containing the reconstruction model settings
        """
        self[Tags.RECONSTRUCTION_MODEL_SETTINGS] = Settings(reconstruction_settings)

    def serialize(self):
        return {"Settings": dict(self)}

    @staticmethod
    def deserialize(dictionary_to_deserialize: dict):
        return Settings(dictionary_to_deserialize, verbose=False)
