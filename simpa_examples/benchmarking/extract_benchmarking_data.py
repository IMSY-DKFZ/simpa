# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from typing import Union


def lines_that_contain(string, fp):
    """
    Function to determine if a string contains a certain word/phrase
    :param string: what to check
    :param fp:
    :return:
    """
    return [line for line in fp if string in line]


def read_out_benchmarking_data(profiles: list = None, start: float = .2, stop: float = .4, step: float = .1,
                               savefolder: str = None) -> None:
    """ Reads benchmarking data and creates a pandas dataframe
    :param profiles: list with profiles ['TIME', "GPU_MEMORY", "MEMORY"]
    :param start: start spacing default .2
    :param stop: stop spacing default .4
    :param step: step size default .1
    :param savefolder: PATH TO benchmarking data txt --> assumes txt: savefolder + "/benchmarking_data_" +
                          profile + "_" + str(spacing) + ".txt"
    :return: None
    :raises: ImportError (unknown units from input data)
    """

    # init defaults
    if savefolder is None or savefolder == "default":
        savefolder = str(Path(__file__).parent.resolve())

    if profiles is None:
        profiles = ['TIME', "GPU_MEMORY", "MEMORY"]

    spacings = np.arange(start, stop+1e-6, step).tolist()

    # specific information for the location of the data in the line profiler files
    info_starts = {"MEMORY": 19, "GPU_MEMORY": 12, "TIME": 16}
    info_ends = {"MEMORY": 29, "GPU_MEMORY": 26, "TIME": 29}

    benchmarking_lists = []  # init result
    for profile in profiles:
        for spacing in spacings:
            file_name = savefolder + "/benchmarking_data_" + profile + "_" + str(np.round(spacing, 4)) + ".txt"
            benchmarking_file = open(file_name, 'r')
            current_examples = []

            # where to find which files have successfully run
            if profile == 'TIME':
                example_name_lines = lines_that_contain("File:", benchmarking_file)

                for enl in example_name_lines:
                    example = enl.rpartition("simpa_examples/")[2].rpartition(".py")[0]
                    if example not in current_examples:
                        current_examples.append(example)
                    else:
                        break

            elif profile == 'GPU_MEMORY':
                example_name_lines = lines_that_contain("##", benchmarking_file)

                for enl in example_name_lines:
                    example = enl.rpartition("run_")[2].rpartition("\n")[0]
                    if example not in current_examples:
                        current_examples.append(example)
                    else:
                        break

            if profile == 'MEMORY':
                example_name_lines = lines_that_contain("Filename:", benchmarking_file)

                for enl in example_name_lines:
                    example = enl.rpartition("simpa_examples/")[2].rpartition(".py")[0]
                    if example not in current_examples:
                        current_examples.append(example)
                    else:
                        break

            benchmarking_file = open(file_name, 'r')
            lines_with_sp_simulate = lines_that_contain("sp.simulate", benchmarking_file)

            examples_counter = 0
            for lwss in lines_with_sp_simulate:
                try:
                    value = float(lwss[info_starts[profile]:info_ends[profile]])
                except ValueError:
                    continue

                unit = str(lwss[info_ends[profile]])
                if profile == "TIME":
                    value_with_unit = value
                else:
                    if unit == 'K':
                        value_with_unit = value / 1000
                    elif unit == 'M':
                        value_with_unit = value
                    elif unit == 'G':
                        value_with_unit = value * 1000
                    else:
                        raise ImportError(f'Unit {unit} not supported')

                example = current_examples[examples_counter]
                benchmarking_lists.append([example, spacing, profile, value_with_unit])

                # lets you know which example you are on
                examples_counter += 1
                if examples_counter == len(current_examples):
                    examples_counter = 0

    # creating data frame
    new_df = pd.DataFrame(benchmarking_lists, columns=['Example', 'Spacing', 'Profile', 'Value'])
    new_df.astype(dtype={"Example": "str", "Spacing": "float64", "Profile": "str", "Value": "float64"})

    # if exists: load old dataframe and append OR just save df
    df_file = Path(savefolder + '/benchmarking_data_frame.csv')
    if df_file.is_file():
        old_df = pd.read_csv(df_file)
        new_df = pd.concat([old_df, new_df])
    new_df.to_csv(df_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run benchmarking tests')
    parser.add_argument("--start", default=.2,
                        help='start spacing default .2mm')
    parser.add_argument("--stop", default=.4,
                        help='stop spacing default .4mm')
    parser.add_argument("--step", default=.1,
                        help='step size mm')
    parser.add_argument("--profiles", default=None, type=str,
                        help='the profile to run')
    parser.add_argument("--savefolder", default=None, type=str, help='where to save the results')
    config = parser.parse_args()

    start = config.start
    stop = config.stop
    step = config.step
    profiles = config.profiles
    savefolder = config.savefolder

    profiles = profiles.split('%')[:-1]
    read_out_benchmarking_data(start=float(start), stop=float(stop), step=float(step), profiles=profiles,
                               savefolder=savefolder)
