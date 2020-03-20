import h5py


def save_hdf5(dictionary, filepath):
    """
    Saves a dictionary with arbitrary content to an hdf5-file with given filepath.
    :param dictionary: Dictionary to save.
    :param filepath: Path of the file to save the dictionary in.
    return
    """

    def data_grabber(file, path, dictionary):
        """
        Helper function which recursively grabs data from dictionaries in order to store them into hdf5 groups.
        :param file: hdf5 file instance to store the data in.
        :param path: Current group path in hdf5 file group structure.
        :param dictionary: Dictionary to save.
        """
        for key, item in dictionary.items():
            if not isinstance(item, (list, dict)):
                h5file[path + key] = item
            elif isinstance(item, list):
                list_dict = dict()
                for i, list_item in enumerate(item):
                    list_dict[str(i)] = list_item
                data_grabber(file, path + key + "/list/", list_dict)
            else:
                data_grabber(file, path + key + "/", item)

    with h5py.File(filepath, "w") as h5file:
        data_grabber(h5file, "/", dictionary)


def load_hdf5(filepath):
    """
    Loads a dictionary from an hdf5 file.
    :param filepath: Path of the file to load the dictionary from.
    :return: Dictionary
    """

    def data_grabber(file, path):
        """
        Helper function which recursively loads data from the hdf5 group structure to a dictionary.
        :param file: hdf5 file instance to load the data from.
        :param path: Current group path in hdf5 file group structure.
        :return: Dictionary
        """
        dictionary = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                dictionary[key] = item[()]
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

    with h5py.File(filepath, "r") as h5file:
        return data_grabber(h5file, "/")
