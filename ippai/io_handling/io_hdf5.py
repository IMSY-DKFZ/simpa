# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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

import h5py
from ippai.utils.serialization import IPPAISerializer
from ippai.utils import AbsorptionSpectrum, Chromophore


def save_hdf5(dictionary: dict, file_path: str, file_dictionary_path: str = "/"):
    """
    Saves a dictionary with arbitrary content to an hdf5-file with given filepath.

    :param dictionary: Dictionary to save.
    :param file_path: Path of the file to save the dictionary in.
    :param file_dictionary_path: Path in dictionary structure of existing hdf5 file to store the dictionary in.
    :returns: :mod:`Null`
    """

    def data_grabber(file, path, data_dictionary):
        """
        Helper function which recursively grabs data from dictionaries in order to store them into hdf5 groups.

        :param file: hdf5 file instance to store the data in.
        :param path: Current group path in hdf5 file group structure.
        :param data_dictionary: Dictionary to save.
        """
        serializer = IPPAISerializer()
        for key, item in data_dictionary.items():
            if not isinstance(item, (list, dict, type(None))):

                if isinstance(item, Chromophore):
                    data_grabber(file, path + key + "/chromophore/", serializer.serialize(item))
                elif isinstance(item, AbsorptionSpectrum):
                    data_grabber(file, path + key + "/absorption_spectrum/", serializer.serialize(item))
                else:
                    try:
                        h5file[path + key] = item
                    except RuntimeError:
                        del h5file[path + key]
                        try:
                            h5file[path + key] = item
                        except RuntimeError as e:
                            print("item", item, "of type", type(item), "was not serializable! Full exception:", e)
                            raise e
            elif item is None:
                h5file[path + key] = "None"
            elif isinstance(item, list):
                list_dict = dict()
                for i, list_item in enumerate(item):
                    list_dict[str(i)] = list_item
                data_grabber(file, path + key + "/list/", list_dict)
            else:
                data_grabber(file, path + key + "/", item)

    if file_dictionary_path == "/":
        writing_mode = "w"
    else:
        writing_mode = "r+"

    with h5py.File(file_path, writing_mode) as h5file:
        data_grabber(h5file, file_dictionary_path, dictionary)


def load_hdf5(file_path, file_dictionary_path="/"):
    """
    Loads a dictionary from an hdf5 file.

    :param file_path: Path of the file to load the dictionary from.
    :param file_dictionary_path: Path in dictionary structure of hdf5 file to lo the dictionary in.
    :returns: Dictionary
    """

    def data_grabber(file, path):
        """
        Helper function which recursively loads data from the hdf5 group structure to a dictionary.

        :param file: hdf5 file instance to load the data from.
        :param path: Current group path in hdf5 file group structure.
        :returns: Dictionary
        """
        dictionary = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                if item[()] is not None:
                    dictionary[key] = item[()]
                else:
                    dictionary[key] = None
            elif isinstance(item, h5py._hl.group.Group):
                if key == "list":
                    dictionary = list()
                    for listkey in sorted(item.keys()):
                        if isinstance(item[listkey], h5py._hl.dataset.Dataset):
                            dictionary.append(item[listkey][()])
                        elif isinstance(item[listkey], h5py._hl.group.Group):
                            dictionary.append(
                                data_grabber(file, path + key + "/" + listkey + "/"))
                else:
                    dictionary[key] = data_grabber(file, path + key + "/")
        return dictionary

    with h5py.File(file_path, "r") as h5file:
        return data_grabber(h5file, file_dictionary_path)