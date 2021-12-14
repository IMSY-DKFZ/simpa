# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import h5py
from simpa.io_handling.serialization import SERIALIZATION_MAP
from simpa.utils.dict_path_manager import generate_dict_path
import numpy as np
from simpa.log import Logger
from simpa.utils.serializer import SerializableSIMPAClass

logger = Logger()


def save_hdf5(save_item, file_path: str, file_dictionary_path: str = "/", file_compression: str = None):
    """
    Saves a dictionary with arbitrary content or an item of any kind to an hdf5-file with given filepath.

    :param save_item: Dictionary to save.
    :param file_path: Path of the file to save the dictionary in.
    :param file_dictionary_path: Path in dictionary structure of existing hdf5 file to store the dictionary in.
    :param file_compression: possible file compression for the hdf5 output file. Values are: gzip, lzf and szip.
    :returns: :mod:`Null`
    """

    def data_grabber(file, path, data_dictionary, compression: str = None):
        """
        Helper function which recursively grabs data from dictionaries in order to store them into hdf5 groups.

        :param file: hdf5 file instance to store the data in.
        :param path: Current group path in hdf5 file group structure.
        :param data_dictionary: Dictionary to save.
        :param compression: possible file compression for the corresponding dataset. Values are: gzip, lzf and szip.
        """

        for key, item in data_dictionary.items():
            key = str(key)
            if isinstance(item, SerializableSIMPAClass):
                serialized_item = item.serialize()

                data_grabber(file, path + key + "/", serialized_item, file_compression)
            elif not isinstance(item, (list, dict, type(None))):

                if isinstance(item, (bytes, int, np.int64, float, str, bool, np.bool_)):
                    try:
                        h5file[path + key] = item
                    except (OSError, RuntimeError, ValueError):
                        del h5file[path + key]
                        h5file[path + key] = item
                else:
                    c = None
                    if isinstance(item, np.ndarray):
                        c = compression

                    try:
                        h5file.create_dataset(path + key, data=item, compression=c)
                    except (OSError, RuntimeError, ValueError):
                        del h5file[path + key]
                        try:
                            h5file.create_dataset(path + key, data=item, compression=c)
                        except RuntimeError as e:
                            logger.critical("item " + str(item) + " of type " + str(type(item)) +
                                            " was not serializable! Full exception: " + str(e))
                            raise e
                    except TypeError as e:
                        logger.critical("The key " + str(key) + " was not of the correct typing for HDF5 handling."
                                        "Make sure this key is not a tuple. " + str(item) + " " + str(type(item)))
                        raise e
            elif item is None:
                try:
                    h5file[path + key] = "None"
                except (OSError, RuntimeError, ValueError):
                    del h5file[path + key]
                    h5file[path + key] = "None"
            elif isinstance(item, list):
                list_dict = dict()
                for i, list_item in enumerate(item):
                    list_dict[str(i)] = list_item
                try:
                    data_grabber(file, path + key + "/list/", list_dict, file_compression)
                except TypeError as e:
                    logger.critical("The key " + str(key) + " was not of the correct typing for HDF5 handling."
                                    "Make sure this key is not a tuple.")
                    raise e
            else:
                data_grabber(file, path + key + "/", item, file_compression)

    if file_dictionary_path == "/":
        writing_mode = "w"
    else:
        writing_mode = "a"

    if isinstance(save_item, SerializableSIMPAClass):
        save_item = save_item.serialize()
    if isinstance(save_item, dict):
        with h5py.File(file_path, writing_mode) as h5file:
            data_grabber(h5file, file_dictionary_path, save_item, file_compression)
    else:
        save_key = file_dictionary_path.split("/")[-2]
        dictionary = {save_key: save_item}
        file_dictionary_path = "/".join(file_dictionary_path.split("/")[:-2]) + "/"
        with h5py.File(file_path, writing_mode) as h5file:
            data_grabber(h5file, file_dictionary_path, dictionary, file_compression)


def load_hdf5(file_path, file_dictionary_path="/"):
    """
    Loads a dictionary from an hdf5 file.

    :param file_path: Path of the file to load the dictionary from.
    :param file_dictionary_path: Path in dictionary structure of hdf5 file to lo the dictionary in.
    :returns: Dictionary
    :rtype: dict
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
                    if isinstance(dictionary[key], bytes):
                        dictionary[key] = dictionary[key].decode("utf-8")
                    elif isinstance(dictionary[key], np.bool_):
                        dictionary[key] = bool(dictionary[key])
                else:
                    dictionary[key] = None
            elif isinstance(item, h5py._hl.group.Group):
                if key in SERIALIZATION_MAP.keys():
                    serialized_dict = data_grabber(file, path + key + "/")
                    serialized_class = SERIALIZATION_MAP[key]
                    deserialized_class = serialized_class.deserialize(serialized_dict)
                    dictionary = deserialized_class
                elif key == "list":
                    dictionary_list = [None for x in item.keys()]
                    for listkey in sorted(item.keys()):
                        if isinstance(item[listkey], h5py._hl.dataset.Dataset):
                            if item[listkey][()] is not None:
                                list_item = item[listkey][()]
                                if isinstance(list_item, bytes):
                                    list_item = list_item.decode("utf-8")
                                elif isinstance(list_item, np.bool_):
                                    list_item = bool(list_item)
                            else:
                                list_item = None
                            dictionary_list[int(listkey)] = list_item
                        elif isinstance(item[listkey], h5py._hl.group.Group):
                            dictionary_list[int(listkey)] = data_grabber(file, path + key + "/" + listkey + "/")
                    dictionary = dictionary_list
                else:
                    dictionary[key] = data_grabber(file, path + key + "/")
        return dictionary

    with h5py.File(file_path, "r") as h5file:
        return data_grabber(h5file, file_dictionary_path)


def load_data_field(file_path, data_field, wavelength=None):
    dict_path = generate_dict_path(data_field, wavelength=wavelength)
    data_field_key = dict_path.split("/")[-2]
    dict_path = "/".join(dict_path.split("/")[:-2]) + "/"
    data = load_hdf5(file_path, dict_path)[data_field_key]
    return data


def save_data_field(data, file_path, data_field, wavelength=None):
    dict_path = generate_dict_path(data_field, wavelength=wavelength)
    save_hdf5(data, file_path, dict_path)
