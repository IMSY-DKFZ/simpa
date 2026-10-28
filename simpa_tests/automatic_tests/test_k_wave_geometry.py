# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import unittest
from unittest.mock import Mock, patch
from simpa import KWaveAdapter, Settings, Tags

class TestKWaveGeometry(unittest.TestCase):

    @staticmethod
    def _settings(spacing_mm=1.0):
        return Settings({
            Tags.SPACING_MM: spacing_mm,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.5,
            Tags.SENSOR_SAMPLING_RATE_MHZ: 10.0,
            Tags.DETECTOR_ELEMENT_WIDTH_MM: 2.0,
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.GPU: False,
        })

    @staticmethod
    def _mock_array(k_wave_array, shape, element_count=1):
        detector_array = Mock()
        detector_array.get_array_binary_mask.return_value = np.ones(shape, dtype=bool)
        detector_array.combine_sensor_data.return_value = np.zeros((element_count, 4))
        k_wave_array.return_value = detector_array
        return detector_array

    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kspaceFirstOrder2D')
    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kWaveArray')
    def test_2d_grid_and_material_shapes(self, k_wave_array, solver):
        detector_array = self._mock_array(k_wave_array, (4, 5), 2)
        solver.return_value = {'p': np.zeros((4, 2)), 'Nt': 4, 'dt': 1e-7}
        initial_pressure = np.ones((4, 5))
        data = {
            Tags.DATA_FIELD_INITIAL_PRESSURE: initial_pressure,
            Tags.DATA_FIELD_SPEED_OF_SOUND: np.full_like(initial_pressure, 1500.0),
            Tags.DATA_FIELD_ALPHA_COEFF: np.full_like(initial_pressure, 0.01),
            Tags.DATA_FIELD_DENSITY: np.full_like(initial_pressure, 1000.0),
            Tags.SENSOR_ELEMENT_POSITIONS: np.array([[-1.0, 4.0], [1.0, 5.0]]),
            Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE: np.zeros(2),
        }

        object.__new__(KWaveAdapter).run_k_wave_simulation(data, self._settings())

        grid = detector_array.get_array_binary_mask.call_args.args[0]
        np.testing.assert_array_equal(grid.N, np.array([4, 5]))
        np.testing.assert_array_equal(
            solver.call_args.args[2].mask,
            detector_array.get_array_binary_mask.return_value,
        )
        medium = solver.call_args.args[3]
        self.assertEqual(medium.sound_speed.shape, (4, 5))
        self.assertEqual(medium.alpha_coeff.shape, (4, 5))
        self.assertEqual(medium.density.shape, (4, 5))

    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kspaceFirstOrder2D')
    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kWaveArray')
    def test_2d_homogeneous_scalar_material_properties(self, k_wave_array, solver):
        """Verify scalar acoustic properties remain scalar and are not passed to np.pad."""
        self._mock_array(k_wave_array, (4, 5))
        solver.return_value = {'p': np.zeros((4, 1)), 'Nt': 4, 'dt': 1e-7}
        data = {
            Tags.DATA_FIELD_INITIAL_PRESSURE: np.ones((4, 5)),
            Tags.SENSOR_ELEMENT_POSITIONS: np.array([[1.0], [1.0]]),
            Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE: np.zeros(1),
        }

        object.__new__(KWaveAdapter).run_k_wave_simulation(data, self._settings())

        medium = solver.call_args.args[3]
        self.assertEqual(float(np.asarray(medium.sound_speed).squeeze()), 1540.0)
        self.assertEqual(medium.alpha_coeff, 0.01)
        self.assertEqual(medium.density, 1000.0)

    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kspaceFirstOrder2D')
    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kWaveArray')
    def test_2d_rectangle_orientation_is_converted_to_degrees(self, k_wave_array, solver):
        """Verify SIMPA's radian orientation is converted to degrees for kWaveArray."""
        detector_array = self._mock_array(k_wave_array, (4, 5))
        solver.return_value = {'p': np.zeros((4, 1)), 'Nt': 4, 'dt': 1e-7}
        material_map = np.ones((4, 5))
        data = {
            Tags.DATA_FIELD_INITIAL_PRESSURE: np.ones((4, 5)),
            Tags.DATA_FIELD_SPEED_OF_SOUND: material_map * 1540.0,
            Tags.DATA_FIELD_ALPHA_COEFF: material_map * 0.01,
            Tags.DATA_FIELD_DENSITY: material_map * 1000.0,
            Tags.SENSOR_ELEMENT_POSITIONS: np.array([[1.0], [1.0]]),
            Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE: np.array([np.pi / 2]),
        }

        object.__new__(KWaveAdapter).run_k_wave_simulation(data, self._settings())

        self.assertAlmostEqual(detector_array.add_rect_element.call_args.args[3], 90.0)

    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kspaceFirstOrder3D')
    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kWaveArray')
    def test_3d_grid_has_no_voxel_offset(self, k_wave_array, solver):
        detector_array = self._mock_array(k_wave_array, (4, 6, 8))
        solver.return_value = {'p': np.zeros((4, 1)), 'Nt': 4, 'dt': 1e-7}
        data = {
            Tags.DATA_FIELD_INITIAL_PRESSURE: np.ones((4, 6, 8)),
            Tags.SENSOR_ELEMENT_POSITIONS: np.array([[2.0], [3.0], [4.0]]),
            Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE: np.zeros((3, 1)),
            Tags.KWAVE_PROPERTY_INTRINSIC_EULER_ANGLE: np.zeros((1, 3)),
        }

        object.__new__(KWaveAdapter).run_k_wave_simulation(data, self._settings())

        np.testing.assert_allclose(
            detector_array.add_rect_element.call_args.args[0],
            np.zeros(3),
            atol=1e-12,
        )

    @staticmethod
    def _solver_result_for_real_sensor_mask(*solver_args):
        sensor = solver_args[2]
        number_sensor_points = int(np.count_nonzero(sensor.mask))
        return {
            'p': np.zeros((4, number_sensor_points)),
            'Nt': 4,
            'dt': 1e-7,
        }

    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kspaceFirstOrder2D')
    def test_2d_boundary_detector_produces_nonempty_sensor_mask(self, solver):
        """Verify a 2D boundary detector remains inside the k-Wave sensor grid."""
        solver.side_effect = self._solver_result_for_real_sensor_mask
        data = {
            Tags.DATA_FIELD_INITIAL_PRESSURE: np.ones((4, 5)),
            Tags.SENSOR_ELEMENT_POSITIONS: np.zeros((2, 1)),
            Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE: np.zeros(1),
        }

        object.__new__(KWaveAdapter).run_k_wave_simulation(data, self._settings())

        sensor = solver.call_args.args[2]
        self.assertGreater(np.count_nonzero(sensor.mask), 0)

    @patch('simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kspaceFirstOrder3D')
    def test_3d_boundary_detector_produces_nonempty_sensor_mask(self, solver):
        """Verify a 3D boundary detector produces a nonempty k-Wave sensor mask."""
        solver.side_effect = self._solver_result_for_real_sensor_mask
        data = {
            Tags.DATA_FIELD_INITIAL_PRESSURE: np.ones((4, 6, 8)),
            Tags.SENSOR_ELEMENT_POSITIONS: np.zeros((3, 1)),
            Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE: np.zeros((3, 1)),
            Tags.KWAVE_PROPERTY_INTRINSIC_EULER_ANGLE: np.zeros((1, 3)),
        }

        object.__new__(KWaveAdapter).run_k_wave_simulation(data, self._settings())

        sensor = solver.call_args.args[2]
        self.assertGreater(np.count_nonzero(sensor.mask), 0)

    @patch("simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kspaceFirstOrder2D")
    @patch("simpa.core.simulation_modules.acoustic_module.k_wave_adapter.kWaveArray")

    def test_detector_position_is_independent_of_resolution(self, k_wave_array, solver):
        def get_position(shape, spacing_mm):
            detector_array = self._mock_array(k_wave_array, shape)
            solver.return_value = {"p": np.zeros((4,1)), "Nt": 4, "dt": 1e-7}

            data = {
                Tags.DATA_FIELD_INITIAL_PRESSURE: np.ones(shape),
                Tags.SENSOR_ELEMENT_POSITIONS: np.array([[2.0], [3.0]]),
                Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE: np.zeros(1),
            }

            object.__new__(KWaveAdapter).run_k_wave_simulation(data, self._settings(spacing_mm))

            return np.asarray(detector_array.add_rect_element.call_args.args[0])

        coarse_position = get_position(shape=(6, 8), spacing_mm=1.0)
        fine_position = get_position(shape=(12, 16), spacing_mm=0.5)

        np.testing.assert_allclose(coarse_position, fine_position, atol=1e-12)

if __name__ == '__main__':
    unittest.main()
