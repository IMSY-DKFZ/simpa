# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np

from simpa import MCXReflectanceAdapter, MCXAdapter, KWaveAdapter, TimeReversalAdapter, Tags, Settings
from simpa.utils.matlab import generate_matlab_cmd


class TestAdditionalFlags(unittest.TestCase):
    def setUp(self) -> None:
        self.additional_flags = ('-l', '-a')
        self.settings = Settings()

    def test_get_cmd_mcx_reflectance_adapter(self):
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
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_BINARY_PATH: '.',
            Tags.ADDITIONAL_FLAGS: self.additional_flags
        })
        mcx_adapter = MCXAdapter(global_settings=self.settings)
        cmd = mcx_adapter.get_command()
        for flag in self.additional_flags:
            self.assertIn(
                flag, cmd, f"{flag} was not in command returned by mcx adapter but was defined as additional flag")

    def test_get_cmd_kwave_adapter(self):
        self.settings.set_acoustic_settings({
            Tags.ADDITIONAL_FLAGS: self.additional_flags
        })
        kwave_adapter = KWaveAdapter(global_settings=self.settings)
        cmd = generate_matlab_cmd("./matlab.exe", "simulate_2D.m", "my_hdf5.mat", kwave_adapter.get_additional_flags())
        for flag in self.additional_flags:
            self.assertIn(
                flag, cmd, f"{flag} was not in command returned by kwave adapter but was defined as additional flag")

    def test_get_cmd_time_reversal_adapter(self):
        self.settings.set_reconstruction_settings({
            Tags.ADDITIONAL_FLAGS: self.additional_flags
        })
        time_reversal_adapter = TimeReversalAdapter(global_settings=self.settings)
        cmd = generate_matlab_cmd("./matlab.exe", "time_reversal_2D.m", "my_hdf5.mat",
                                  time_reversal_adapter.get_additional_flags())
        for flag in self.additional_flags:
            self.assertIn(
                flag, cmd, f"{flag} was not in command returned by time reversal adapter but was defined as additional flag")


if __name__ == '__main__':
    unittest.main()
