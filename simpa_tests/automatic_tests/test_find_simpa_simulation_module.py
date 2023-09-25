# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import os


class TestFindSimulationModule(unittest.TestCase):

    def test_find_simulation_module(self):
        simulation_module = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                         'simpa', 'core', 'simulation_modules')
        assert os.path.exists(simulation_module), 'Could not find simulation_module'
        simulate_2D_path = os.path.join(simulation_module, 'acoustic_forward_module', 'simulate_2D.m')
        assert os.path.exists(simulate_2D_path), 'Could not find simulate_2D.m'
        simulate_2D_path = os.path.join(simulation_module, 'reconstruction_module', 'time_reversal_2D.m')
        assert os.path.exists(simulate_2D_path), 'Could not find time_reversal_2D.m'