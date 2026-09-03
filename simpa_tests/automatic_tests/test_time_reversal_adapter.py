# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import unittest
from unittest.mock import Mock, patch

import numpy as np

from simpa import LinearArrayDetectionGeometry, Settings, Tags, TimeReversalAdapter


class TestTimeReversalAdapter(unittest.TestCase):

    @staticmethod
    def _adapter_for_sensor_map():
        adapter = object.__new__(TimeReversalAdapter)
        adapter.component_settings = Settings({
            Tags.SPACING_MM: 1.0,
            Tags.ACOUSTIC_SIMULATION_3D: False,
        })
        adapter.global_settings = Settings({
            Tags.DIM_VOLUME_X_MM: 10.0,
            Tags.DIM_VOLUME_Y_MM: 1.0,
            Tags.DIM_VOLUME_Z_MM: 10.0,
        })
        adapter.logger = Mock()
        return adapter

    @patch('simpa.core.simulation_modules.reconstruction_module.time_reversal_adapter.reorder_sensor_data')
    @patch('simpa.core.simulation_modules.reconstruction_module.time_reversal_adapter.kspaceFirstOrder2D')
    def test_2d_solver_handoff(self, solver, reorder_sensor_data):
        """Verify grid, mask, detector rows, options, and output are handed off intact."""
        sensor_mask = np.zeros((5, 7), dtype=bool)
        sensor_mask[0, [1, 3, 5]] = True
        time_series_data = np.arange(12, dtype=float).reshape(3, 4)
        reorder_sensor_data.return_value = np.array([[2], [0], [1]])
        expected_reconstruction = np.arange(35, dtype=float).reshape(5, 7)
        solver.return_value = {'p_final': expected_reconstruction}
        data = {
            Tags.KWAVE_PROPERTY_SENSOR_MASK: sensor_mask,
            Tags.DATA_FIELD_TIME_SERIES_DATA: time_series_data,
        }
        settings = Settings({
            Tags.SPACING_MM: 1.0,
            Tags.K_WAVE_SPECIFIC_NT: 4,
            Tags.K_WAVE_SPECIFIC_DT: 1e-7,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.5,
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.GPU: False,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: r'C:\custom\kspaceFirstOrder-OMP.exe',
            Tags.ADDITIONAL_FLAGS: ('--verbose', '2', '--show-sim-log', 'false'),
        })

        reconstruction = object.__new__(TimeReversalAdapter).run_k_wave_timereversal(data, settings)
        grid, _, sensor, _, _, execution_options = solver.call_args.args

        np.testing.assert_array_equal(grid.N, np.array([5, 7]))
        self.assertEqual(grid.Nt, 4)
        self.assertAlmostEqual(grid.dt, 1e-7)
        np.testing.assert_array_equal(sensor.mask, sensor_mask)
        np.testing.assert_array_equal(sensor.time_reversal_boundary_data, time_series_data[[1, 2, 0]])
        self.assertFalse(execution_options.is_gpu_simulation)
        self.assertEqual(execution_options.verbose_level, 2)
        self.assertFalse(execution_options.show_sim_log)
        self.assertEqual(
            str(execution_options._binary_path),
            r'C:\custom\kspaceFirstOrder-OMP.exe',
        )
        np.testing.assert_array_equal(reconstruction, expected_reconstruction)

    @patch('simpa.core.simulation_modules.reconstruction_module.time_reversal_adapter.kspaceFirstOrder2D')
    def test_2d_solver_uses_inverse_of_real_kwave_angular_order(self, solver):
        """Verify SIMPA detector rows use the inverse k-Wave angular permutation."""
        sensor_mask = np.zeros((7, 7), dtype=bool)
        sensor_mask[0, 0:3] = True
        sensor_mask[2, 1] = True
        time_series_data = np.arange(16, dtype=float).reshape(4, 4)
        solver.return_value = {'p_final': np.zeros((7, 7))}
        data = {
            Tags.KWAVE_PROPERTY_SENSOR_MASK: sensor_mask,
            Tags.DATA_FIELD_TIME_SERIES_DATA: time_series_data,
        }
        settings = Settings({
            Tags.SPACING_MM: 1.0,
            Tags.K_WAVE_SPECIFIC_NT: 4,
            Tags.K_WAVE_SPECIFIC_DT: 1e-7,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.5,
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.GPU: False,
        })

        object.__new__(TimeReversalAdapter).run_k_wave_timereversal(data, settings)
        sensor = solver.call_args.args[2]

        # For this real mask, k-wave-python's angular order is [2, 0, 1, 3]
        # and its inverse is [1, 2, 0, 3]. These are deliberately different.
        np.testing.assert_array_equal(
            sensor.time_reversal_boundary_data,
            time_series_data[[1, 2, 0, 3]],
        )

    @patch('simpa.core.simulation_modules.reconstruction_module.time_reversal_adapter.reorder_sensor_data')
    @patch('simpa.core.simulation_modules.reconstruction_module.time_reversal_adapter.kspaceFirstOrder2D')
    def test_2d_solver_rejects_invalid_angular_permutation(self, solver, reorder_sensor_data):
        """Verify reconstruction rejects a sensor mapping that is not a permutation."""
        sensor_mask = np.zeros((5, 7), dtype=bool)
        sensor_mask[0, [1, 3, 5]] = True
        reorder_sensor_data.return_value = np.array([[0], [0], [2]])
        data = {
            Tags.KWAVE_PROPERTY_SENSOR_MASK: sensor_mask,
            Tags.DATA_FIELD_TIME_SERIES_DATA: np.zeros((3, 4)),
        }
        settings = Settings({
            Tags.SPACING_MM: 1.0,
            Tags.K_WAVE_SPECIFIC_NT: 4,
            Tags.K_WAVE_SPECIFIC_DT: 1e-7,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.5,
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.GPU: False,
        })

        with self.assertRaisesRegex(ValueError, 'same detectors'):
            object.__new__(TimeReversalAdapter).run_k_wave_timereversal(data, settings)

        solver.assert_not_called()

    def test_sensor_map_uses_zero_based_python_coordinates(self):
        """Verify NumPy sensor masks do not inherit MATLAB's one-based index shift."""
        adapter = self._adapter_for_sensor_map()
        geometry = Mock(number_detector_elements=3)
        geometry.get_detector_element_positions_accounting_for_device_position_mm.return_value = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 5.0],
            [8.5, 0.0, 8.5],
        ])

        input_data, spacing_mm = adapter.get_acoustic_properties({}, geometry)
        sensor_map = input_data[Tags.KWAVE_PROPERTY_SENSOR_MASK]

        self.assertEqual(sensor_map[0, 0], 1)
        self.assertEqual(sensor_map[5, 5], 1)
        self.assertEqual(sensor_map[9, 9], 1)
        self.assertEqual(np.count_nonzero(sensor_map), 3)
        self.assertEqual(sensor_map.shape, (11, 11))
        self.assertEqual(spacing_mm, 1.0)

    def test_sensor_map_rejects_negative_out_of_grid_position(self):
        """Verify negative detector coordinates raise instead of wrapping in NumPy."""
        adapter = self._adapter_for_sensor_map()
        geometry = Mock(number_detector_elements=1)
        geometry.get_detector_element_positions_accounting_for_device_position_mm.return_value = np.array([
            [-1.0, 0.0, 5.0],
        ])

        with self.assertRaisesRegex(ValueError, 'outside the reconstruction grid'):
            adapter.get_acoustic_properties({}, geometry)

    @patch('simpa.core.simulation_modules.reconstruction_module.time_reversal_adapter.compute_image_dimensions')
    def test_reconstruction_preserves_solver_axis_order(self, compute_image_dimensions):
        """Verify reconstruction preserves k-wave-python's native axis order."""
        adapter = object.__new__(TimeReversalAdapter)
        adapter.component_settings = Settings({Tags.ACOUSTIC_SIMULATION_3D: False})
        adapter.global_settings = Settings({
            Tags.K_WAVE_SPECIFIC_DT: 1e-7,
            Tags.K_WAVE_SPECIFIC_NT: 4,
        })
        adapter.logger = Mock()
        adapter.get_acoustic_properties = Mock(return_value=({}, 1.0))
        solver_native_depth_x = np.arange(12).reshape(3, 4)
        adapter.run_k_wave_timereversal = Mock(return_value=solver_native_depth_x)
        compute_image_dimensions.return_value = (4, 1, 3, 0, 4, 0, 3, 0, 0)
        geometry = LinearArrayDetectionGeometry(
            number_detector_elements=1,
            field_of_view_extent_mm=np.array([0, 4, 0, 0, 0, 3]),
        )

        reconstruction = adapter.reconstruction_algorithm(np.zeros((1, 4)), geometry)

        np.testing.assert_array_equal(reconstruction, solver_native_depth_x[:, :3])


if __name__ == '__main__':
    unittest.main()
