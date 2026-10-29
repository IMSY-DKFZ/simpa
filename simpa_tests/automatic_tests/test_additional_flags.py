# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path
from unittest.mock import patch

from simpa import MCXReflectanceAdapter, MCXAdapter, KWaveAdapter, TimeReversalAdapter, Tags, Settings


class TestAdditionalFlags(unittest.TestCase):
    def setUp(self) -> None:
        self.additional_flags = ('-l', '-a')
        self.settings = Settings()

    def test_get_cmd_mcx_reflectance_adapter(self):
        """Verify MCX reflectance commands include every configured additional flag."""
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_BINARY_PATH: '.',
            Tags.ADDITIONAL_FLAGS: self.additional_flags
        })
        mcx_reflectance_adapter = MCXReflectanceAdapter(global_settings=self.settings)
        cmd = mcx_reflectance_adapter.get_command()
        for flag in self.additional_flags:
            self.assertIn(
                flag, cmd, f"{flag} was not in command returned by mcx reflectance adapter but was defined as additional flag")

    def test_get_cmd_mcx_adapter(self):
        """Verify MCX forward commands include every configured additional flag."""
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_BINARY_PATH: '.',
            Tags.ADDITIONAL_FLAGS: self.additional_flags
        })
        mcx_adapter = MCXAdapter(global_settings=self.settings)
        cmd = mcx_adapter.get_command()
        for flag in self.additional_flags:
            self.assertIn(
                flag, cmd, f"{flag} was not in command returned by mcx adapter but was defined as additional flag")

    def test_k_wave_additional_flags_are_translated(self):
        """Verify command-style flags map to typed k-wave-python execution options."""
        flags = (
            '--verbose', '2',
            '--device-num', '1',
            '--num-threads', '4',
            '--thread-binding', 'true',
            '--show-sim-log', 'false',
        )

        execution_kwargs = KWaveAdapter._translate_additional_flags_to_execution_kwargs(flags)

        self.assertEqual(execution_kwargs, {
            'verbose_level': 2,
            'device_num': 1,
            'num_threads': 4,
            'thread_binding': True,
            'show_sim_log': False,
        })

    def test_k_wave_additional_flags_are_forwarded_to_execution_options(self):
        """Verify translated flags are forwarded to SimulationExecutionOptions."""
        with patch(
                'simpa.core.simulation_modules.acoustic_module.k_wave_adapter.SimulationExecutionOptions'
        ) as execution_options:
            KWaveAdapter._create_execution_options(
                is_gpu_simulation=True,
                additional_flags=('--verbose', '2', '--device-num', '1'),
            )

        execution_options.assert_called_once_with(
            is_gpu_simulation=True,
            verbose_level=2,
            device_num=1,
        )

    def test_k_wave_binary_path_is_forwarded_as_path_object(self):
        """Verify the configured executable is forwarded as a pathlib Path."""
        configured_path = r'C:\custom\kspaceFirstOrder-OMP.exe'

        with patch(
                'simpa.core.simulation_modules.acoustic_module.k_wave_adapter.SimulationExecutionOptions'
        ) as execution_options:
            KWaveAdapter._create_execution_options(
                is_gpu_simulation=False,
                binary_path=configured_path,
            )

        execution_options.assert_called_once_with(
            is_gpu_simulation=False,
            binary_path=Path(configured_path),
        )

    def test_k_wave_selects_cpu_binary(self):
        """Verify GPU=False selects the OMP executable from a bundled binary pair."""
        configured_path = Path('kwave') / 'kspaceFirstOrder-CUDA.exe'

        with patch(
                'simpa.core.simulation_modules.acoustic_module.k_wave_adapter.SimulationExecutionOptions'
        ) as execution_options:
            KWaveAdapter._create_execution_options(
                is_gpu_simulation=False,
                binary_path=configured_path,
            )

        execution_options.assert_called_once_with(
            is_gpu_simulation=False,
            binary_path=Path('kwave') / 'kspaceFirstOrder-OMP.exe',
        )

    def test_k_wave_selects_gpu_binary(self):
        """Verify GPU=True selects the CUDA executable from a bundled binary pair."""
        configured_path = Path('kwave') / 'kspaceFirstOrder-OMP.exe'

        with patch(
                'simpa.core.simulation_modules.acoustic_module.k_wave_adapter.SimulationExecutionOptions'
        ) as execution_options:
            KWaveAdapter._create_execution_options(
                is_gpu_simulation=True,
                binary_path=configured_path,
            )

        execution_options.assert_called_once_with(
            is_gpu_simulation=True,
            binary_path=Path('kwave') / 'kspaceFirstOrder-CUDA.exe',
        )


if __name__ == '__main__':
    unittest.main()
