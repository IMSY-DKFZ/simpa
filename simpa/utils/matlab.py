# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import inspect
import os
from typing import List


def generate_matlab_cmd(matlab_binary_path: str, simulation_script_path: str, data_path: str, additional_flags: List[str] = []) -> List[str]:
    """Generates the MATLAB execution command from the given paths

    :param matlab_binary_path: path to the MATLAB binary file as defined by PathManager
    :type matlab_binary_path: str
    :param simulation_script_path: path to the MATLAB script that should be run (either simulate_2D.m or simulate_3D.m)
    :type simulation_script_path: str
    :param data_path: path to the .mat file used for simulating
    :type data_path: str
    :param additional_flags: list of optional additional flags for MATLAB
    :type additional_flags: List[str]
    :return: list of command parts
    :rtype: List[str]
    """

    # get path of calling script to add to matlab path
    base_script_path = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    # ensure data path is an absolute path
    data_path = os.path.abspath(data_path)

    cmd = list()
    cmd.append(matlab_binary_path)
    cmd.append("-nodisplay")
    cmd.append("-nosplash")
    cmd.append("-automation")
    cmd.append("-wait")
    cmd += additional_flags
    cmd.append("-r")
    cmd.append(f"addpath('{base_script_path}');{simulation_script_path}('{data_path}');exit;")
    return cmd
