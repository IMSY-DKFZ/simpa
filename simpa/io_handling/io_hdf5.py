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
import h5py
from simpa.io_handling.serialization import SIMPASerializer
from simpa.utils import AbsorptionSpectrum, Molecule
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.dict_path_manager import generate_dict_path
import numpy as np

MOLECULE_COMPOSITION = Tags.MOLECULE_COMPOSITION[0]
MOLECULE = "molecule"
ABSORPTION_SPECTRUM = "absorption_spectrum"


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
        serializer = SIMPASerializer()
        for key, item in data_dictionary.items():
            key = str(key)
            if not isinstance(item, (list, dict, type(None))):

                if isinstance(item, Molecule):
                    data_grabber(file, path + key + "/" + MOLECULE + "/",
                                 serializer.serialize(item), file_compression)
                elif isinstance(item, AbsorptionSpectrum):
                    data_grabber(file, path + key + "/" + ABSORPTION_SPECTRUM + "/",
                                 serializer.serialize(item), file_compression)
                else:
                    if isinstance(item, (bytes, int, np.int64, float, str, bool, np.bool_)):
                        try:
                            h5file[path + key] = item
                        except (OSError, RuntimeError, ValueError):
                            del h5file[path + key]
                            h5file[path + key] = item
                    else:
                        try:
                            h5file.create_dataset(path + key, data=item, compression=compression)
                        except (OSError, RuntimeError, ValueError):
                            del h5file[path + key]
                            try:
                                h5file.create_dataset(path + key, data=item, compression=compression)
                            except RuntimeError as e:
                                print("item", item, "of type", type(item), "was not serializable! Full exception:", e)
                                raise e
                        except TypeError as e:
                            print("The key", key, "was not of the correct typing for HDF5 handling."
                                  "Make sure this key is not a tuple.", item, type(item))
                            raise e
            elif item is None:
                h5file[path + key] = "None"
            elif isinstance(item, list):
                list_dict = dict()
                for i, list_item in enumerate(item):
                    list_dict[str(i)] = list_item
                try:
                    data_grabber(file, path + key + "/list/", list_dict, file_compression)
                except TypeError as e:
                    print("The key", key, "was not of the correct typing for HDF5 handling."
                                          "Make sure this key is not a tuple.")
                    raise e
            else:
                data_grabber(file, path + key + "/", item, file_compression)

    if file_dictionary_path == "/":
        writing_mode = "w"
    else:
        writing_mode = "a"

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
                else:
                    dictionary[key] = None
            elif isinstance(item, h5py._hl.group.Group):
                if key == "list":
                    dictionary_list = [None for x in item.keys()]
                    for listkey in sorted(item.keys()):
                        if isinstance(item[listkey], h5py._hl.dataset.Dataset):
                            dictionary_list[int(listkey)] = item[listkey][()]
                        elif isinstance(item[listkey], h5py._hl.group.Group):
                            dictionary_list[int(listkey)] = data_grabber(file, path + key + "/" + listkey + "/")
                    dictionary = dictionary_list
                elif key == MOLECULE_COMPOSITION:
                    mc = MolecularComposition()
                    molecules = data_grabber(file, path + key + "/")
                    for molecule in molecules:
                        mc.append(molecule)
                    dictionary[key] = mc
                elif key == MOLECULE:
                    data = data_grabber(file, path + key + "/")
                    dictionary = Molecule.from_settings(data)
                elif key == ABSORPTION_SPECTRUM:
                    data = data_grabber(file, path + key + "/")
                    dictionary = AbsorptionSpectrum.from_settings(data)
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
