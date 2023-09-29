# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import inspect
import os
from typing import List


def generate_matlab_cmd(matlab_binary_path: str, simulation_script_path: str, data_path: str) -> List[str]:
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
    cmd.append("-r")
    cmd.append(f"addpath('{base_script_path}');{simulation_script_path}('{data_path}');exit;")
    return cmd
