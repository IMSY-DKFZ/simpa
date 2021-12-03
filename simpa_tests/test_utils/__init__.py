# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from simpa.utils import TISSUE_LIBRARY
from simpa.utils.libraries.structure_library import Background, CircularTubularStructure
from simpa.utils.settings import Settings
from simpa.utils import Tags


def assert_equals_recursive(a, b):
    assert isinstance(a, type(b))
    assert isinstance(b, type(a))
    if isinstance(a, dict):
        for item in a:
            assert item in a, (str(item) + " was not in a: " + str(a))
            assert item in b, (str(item) + " was not in b: " + str(b))
            if isinstance(a[item], dict):
                assert_equals_recursive(a[item], b[item])
            elif isinstance(a[item], list):
                assert_equals_recursive(a[item], b[item])
            else:
                if isinstance(a[item], np.ndarray):
                    assert (a[item] == b[item]).all()
                else:
                    assert a[item] == b[item], str(a[item]) + " is not the same as " + str(b[item])
                    if (np.issubdtype(type(a[item]), int) or np.issubdtype(type(a[item]), float)) and \
                            (np.issubdtype(type(b[item]), int) or np.issubdtype(type(b[item]), float)):
                        assert (np.issubdtype(type(a[item]), int) & np.issubdtype(type(b[item]), int)) or \
                               (np.issubdtype(type(a[item]), float) & np.issubdtype(type(b[item]), float))
                    else:
                        assert isinstance(a[item], type(b[item]))
                        assert isinstance(b[item], type(a[item]))

    elif isinstance(a, list):
        for item1, item2 in zip(a, b):
            assert_equals_recursive(item1, item2)
    else:
        assert a == b, str(a) + " is not the same as " + str(b)


def create_background():
    background_structure_dictionary = dict()
    background_structure_dictionary[Tags.PRIORITY] = 0
    background_structure_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    background_structure_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
    return background_structure_dictionary


def create_vessel():
    tubular_structure_dictionary = dict()
    tubular_structure_dictionary[Tags.PRIORITY] = 2
    tubular_structure_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    tubular_structure_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    tubular_structure_dictionary[Tags.STRUCTURE_END_MM] = [10, 10, 10]
    tubular_structure_dictionary[Tags.STRUCTURE_RADIUS_MM] = 4
    tubular_structure_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    tubular_structure_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    tubular_structure_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
    return tubular_structure_dictionary


def create_test_structure_parameters():
    structures_dict = dict()
    structures_dict["background"] = create_background()
    structures_dict["vessel"] = create_vessel()
    return structures_dict


def create_background_of_molecule(global_settings, molecule):
    background_structure_dictionary = dict()
    background_structure_dictionary[Tags.PRIORITY] = 0
    background_structure_dictionary[Tags.MOLECULE_COMPOSITION] = molecule
    bg = Background(global_settings, Settings(background_structure_dictionary))
    return bg.to_settings()


def create_vessel_of_molecule(global_settings, molecule, prio, structure_start):
    tubular_structure_dictionary = dict()
    tubular_structure_dictionary[Tags.PRIORITY] = prio
    tubular_structure_dictionary[Tags.MOLECULE_COMPOSITION] = molecule
    tubular_structure_dictionary[Tags.STRUCTURE_START_MM] = [structure_start, 0, 1]
    tubular_structure_dictionary[Tags.STRUCTURE_END_MM] = [structure_start, 2, 1]
    tubular_structure_dictionary[Tags.STRUCTURE_RADIUS_MM] = 0.5
    tubular_structure_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    tubular_structure_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    tube = CircularTubularStructure(global_settings, Settings(tubular_structure_dictionary))
    return tube.to_settings()


def create_test_structure_of_molecule(global_settings, molecule1, molecule2, molecule3, key):
    structures_dict = dict()
    if key == "setting1":
        structures_dict["background"] = create_background_of_molecule(global_settings, molecule1)
    if key == "setting2":
        structures_dict["background"] = create_background_of_molecule(global_settings, molecule1)
        structures_dict["vessel"] = create_vessel_of_molecule(global_settings, molecule2, prio=1, structure_start=1)
    if key == "setting3":
        structures_dict["background"] = create_background_of_molecule(global_settings, molecule1)
        structures_dict["vessel"] = create_vessel_of_molecule(global_settings, molecule2, prio=1, structure_start=1)
        structures_dict["vessel2"] = create_vessel_of_molecule(global_settings, molecule3, prio=2, structure_start=1.25)
    return structures_dict
