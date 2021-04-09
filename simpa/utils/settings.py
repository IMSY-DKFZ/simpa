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

from simpa.utils import Tags
from simpa.log import Logger


class Settings(dict):
    """
    The Settings class is a dictionary that contains all relevant settings for running a simulation in the SIMPA
    toolkit. It includes an automatic sanity check for input parameters using the simpa.utils.Tags class. \n
    Usage: Seetings({Tags.KEY1: value1, Tags.KEY2: value2, ...})
    """

    def __init__(self, dictionary: dict = None):
        super(Settings, self).__init__()
        self.logger = Logger()
        if dictionary is None:
            dictionary = {}
        for key, value in dictionary.items():
            self[key] = value

    def __setitem__(self, key, value):
        if isinstance(key, str):
            super().__setitem__(key, value)
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
        return self[Tags.OPTICAL_MODEL_SETTINGS]

    def set_optical_settings(self, optical_settings: dict):
        """
        Replaces the currently stored optical settings with the given dictionary

        :param optical_settings: a dictionary containing the optical settings
        """
        self[Tags.OPTICAL_MODEL_SETTINGS] = optical_settings

    def get_volume_creation_settings(self):
        """"
        Returns the settings for the optical forward model that are saved in this settings dictionary
        """
        return self[Tags.VOLUME_CREATION_MODEL_SETTINGS]

    def set_volume_creation_settings(self, volume_settings: dict):
        """
        Replaces the currently stored volume creation settings with the given dictionary

        :param volume_settings: a dictionary containing the volume creator settings
        """
        self[Tags.VOLUME_CREATION_MODEL_SETTINGS] = volume_settings

    def get_acoustic_settings(self):
        """"
        Returns the settings for the acoustic forward model that are saved in this settings dictionary
        """
        return self[Tags.VOLUME_CREATION_MODEL_SETTINGS]

    def set_acoustic_settings(self, acoustic_settings: dict):
        """
        Replaces the currently stored acoustic forward model settings with the given dictionary

        :param acoustic_settings: a dictionary containing the acoustic model settings
        """
        self[Tags.VOLUME_CREATION_MODEL_SETTINGS] = acoustic_settings

    def get_reconstruction_settings(self):
        """"
        Returns the settings for the reconstruction model that are saved in this settings dictionary
        """
        return self[Tags.RECONSTRUCTION_MODEL_SETTINGS]

    def set_reconstruction_settings(self, reconstruction_settings: dict):
        """
        Replaces the currently stored reconstruction model settings with the given dictionary

        :param reconstruction_settings: a dictionary containing the reconstruction model settings
        """
        self[Tags.RECONSTRUCTION_MODEL_SETTINGS] = reconstruction_settings

    def save(self, path):
        from simpa.io_handling.io_hdf5 import save_hdf5
        save_hdf5(self, path)

    def load(self, path):
        from simpa.io_handling.io_hdf5 import load_hdf5
        for key, value in load_hdf5(path).items():
            self[key] = value
