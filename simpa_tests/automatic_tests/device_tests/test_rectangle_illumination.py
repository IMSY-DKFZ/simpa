# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
from simpa import Settings, Tags
from simpa.core.device_digital_twins import RectangleIlluminationGeometry


class TestRectangleIlluminationGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.test_object = RectangleIlluminationGeometry(length_mm=15, width_mm=12)

    def test_if_constructor_is_called_with_invalid_arguments_error_is_raised(self):
        with self.assertRaises(TypeError):
            RectangleIlluminationGeometry(length_mm=None, width_mm=12)
        with self.assertRaises(TypeError):
            RectangleIlluminationGeometry(length_mm=None, width_mm=None)
        with self.assertRaises(AssertionError):
            RectangleIlluminationGeometry(length_mm=0, width_mm=12)
        with self.assertRaises(AssertionError):
            RectangleIlluminationGeometry(length_mm=15, width_mm=0)

    def test_if_get_mcx_illuminator_definition_is_called_with_invalid_arguments_error_is_raised(self):
        with self.assertRaises(AssertionError):
            self.test_object.get_mcx_illuminator_definition(None)

    def test_if_get_mcx_illuminator_definition_is_called_correct_parameters_are_returned(self):
        # Arrange
        global_settings = Settings()
        global_settings[Tags.SPACING_MM] = 0.2
        expected_dict = {
            "Type": Tags.ILLUMINATION_TYPE_PLANAR,
            "Pos": [0, 0, 0],
            "Dir": [0, 0, 1],
            "Param1": [60, 0, 0],
            "Param2": [0, 75, 0]
        }

        # Act
        actual_dict = self.test_object.get_mcx_illuminator_definition(global_settings)

        # Assert
        assert expected_dict == actual_dict

    def test_serialize_and_deserialize_are_inverse_to_each_other(self):
        # Act
        serialized_dict = self.test_object.serialize()
        assert len(serialized_dict.keys()) == 1

        class_name = list(serialized_dict.keys())[0]
        deserialized_object = globals()[class_name].deserialize(serialized_dict[class_name])

        # Assert
        assert set(self.test_object.__dict__.keys()) == set(deserialized_object.__dict__.keys())
        for key, value in deserialized_object.__dict__.items():
            value2 = self.test_object.__dict__[key]

            if isinstance(value, np.ndarray):
                assert np.array_equal(value, value2)
            else:
                assert value == value2

    def test_if_deserialize_is_called_with_invalid_arguments_error_is_raised(self):
        with self.assertRaises(AssertionError):
            RectangleIlluminationGeometry.deserialize(None)
