# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import os

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def lines_that_contain(string, fp):
    """
    Function to determine if a string contains a certain word/phrase and the returns the full line
    :param string: string the line in the fp is compared to
    :param fp: full page of text from profiler output
    :return: lines in the full page that contain 'string'
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
        savefolder = Path(__file__).parent.resolve()
    else:
        savefolder = Path(savefolder)

    if profiles is None:
        profiles = ['TIME', "GPU_MEMORY", "MEMORY"]

    spacings = np.arange(start, stop+1e-6, step).tolist()

    # specific information for the location of the data in the line profiler files
    info_starts = {"MEMORY": 19, "GPU_MEMORY": 12, "TIME": 16}
    info_ends = {"MEMORY": 29, "GPU_MEMORY": 26, "TIME": 29}

    profiling_strings = {"TIME": ("File:", "simpa_examples/", ".py"),
                         "GPU_MEMORY": ("##", "run_", "\n"),
                         "MEMORY": ("Filename:", "simpa_examples/", ".py"), }

    benchmarking_lists = []  # init result
    for profile in profiles:
        for spacing in spacings:
            txt_file = "benchmarking_data_" + profile + "_" + str(np.round(spacing, 4)) + ".txt"
            file_name = savefolder / txt_file
            benchmarking_file = open(file_name, 'r')
            current_examples = []

            # where to find which files have successfully run
            example_name_lines = lines_that_contain(profiling_strings[profile][0], benchmarking_file)
            for enl in example_name_lines:
                example = enl.rpartition(profiling_strings[profile][1])[2].rpartition(profiling_strings[profile][2])[0]
                if example not in current_examples:
                    current_examples.append(example)
                else:
                    break

            with open(file_name, 'r') as benchmarking_file:
                lines_with_sp_simulate = lines_that_contain("sp.simulate", benchmarking_file)

            for e_idx, example in enumerate(current_examples):
                try:
                    value = float(lines_with_sp_simulate[e_idx][info_starts[profile]:info_ends[profile]])
                    unit = str(lines_with_sp_simulate[e_idx][info_ends[profile]])
                except ValueError:
                    with open(file_name, 'r') as benchmarking_file:
                        lines_with_run_sim = lines_that_contain("= run_sim", benchmarking_file)
                    value = 0
                    for line in lines_with_run_sim:
                        value += float(line[info_starts[profile]:info_ends[profile]])
                    unit = str(line[info_ends[profile]])

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

                benchmarking_lists.append([example, spacing, profile, value_with_unit])

    # creating data frame
    new_df = pd.DataFrame(benchmarking_lists, columns=['Example', 'Spacing', 'Profile', 'Value'])
    new_df.astype(dtype={"Example": "str", "Spacing": "float64", "Profile": "str", "Value": "float64"})

    # if exists: remove old file
    df_file = savefolder / 'benchmarking_data_frame.csv'
    if df_file.is_file():
        os.remove(df_file)
    new_df.to_csv(df_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run benchmarking tests')
    parser.add_argument("--start", default=.15,
                        help='start spacing default .2mm')
    parser.add_argument("--stop", default=.25,
                        help='stop spacing default .4mm')
    parser.add_argument("--step", default=.05,
                        help='step size mm')
    parser.add_argument("--profiles", default=None, type=str,
                        help='the profile to run')
    parser.add_argument("--savefolder", default=None, type=str, help='where to save the results')
    config = parser.parse_args()

    profiles = config.profiles
    if profiles and "%" in profiles:
        profiles = profiles.split('%')[:-1]
    elif profiles:
        profiles = [profiles]
    read_out_benchmarking_data(start=float(config.start), stop=float(config.stop), step=float(config.step),
                               profiles=profiles, savefolder=config.savefolder)
