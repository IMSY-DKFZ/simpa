# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import TISSUE_LIBRARY
from simpa_tests.test_utils.tissue_composition_tests import compare_molecular_composition_against_expected_values, \
    get_epidermis_reference_dictionary, get_dermis_reference_dictionary, get_muscle_reference_dictionary, \
    get_fully_oxygenated_blood_reference_dictionary, get_fully_deoxygenated_blood_reference_dictionary

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VISUALISE = False

class TestEpidermis(unittest.TestCase):

    def setUp(self) -> None:
        print("set_up")

    def tearDown(self) -> None:
        print("tear_down")

    def test_epidermis_parameters(self):
        compare_molecular_composition_against_expected_values(
            molecular_composition=TISSUE_LIBRARY.epidermis(0.014),
            expected_values=get_epidermis_reference_dictionary(),
            visualise_values=VISUALISE,
            title="EPIDERMIS"
        )

    def test_dermis_parameters(self):
        compare_molecular_composition_against_expected_values(
            molecular_composition=TISSUE_LIBRARY.dermis(),
            expected_values=get_dermis_reference_dictionary(),
            visualise_values=VISUALISE,
            title="DERMIS"
        )

    def test_muscle_parameters(self):
        compare_molecular_composition_against_expected_values(
            molecular_composition=TISSUE_LIBRARY.muscle(),
            expected_values=get_muscle_reference_dictionary(),
            visualise_values=VISUALISE,
            title="MUSCLE"
        )

    def test_blood_oxy_parameters(self):
        compare_molecular_composition_against_expected_values(
            molecular_composition=TISSUE_LIBRARY.blood(1.0),
            expected_values=get_fully_oxygenated_blood_reference_dictionary(only_use_NIR_values=True),
            visualise_values=VISUALISE,
            title="OXY BLOOD"
        )

    def test_blood_deoxy_parameters(self):
        compare_molecular_composition_against_expected_values(
            molecular_composition=TISSUE_LIBRARY.blood(0.0),
            expected_values=get_fully_deoxygenated_blood_reference_dictionary(only_use_NIR_values=True),
            visualise_values=VISUALISE,
            title="DEOXY BLOOD"
        )
