# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import pathlib
import os
from argparse import ArgumentParser


def run_benchmarking_tests(spacing=0.4, profile: str = "TIME", savefolder: str = 'default'):
    """

    :param spacing: Simulation spacing in mm
    :param profile: What tag to choose to benchmark the examples
    :param savefolder: where to save the results
    :return: file with benchmarking data line by line
    """
    spacing = float(spacing)
    os.environ["SIMPA_PROFILE"] = profile
    if savefolder == 'default':
        savefolder = (str(pathlib.Path(__file__).parent.resolve()) + "/benchmarking_data_" + profile + "_"
                      + str(spacing) + ".txt")
        os.environ["SIMPA_PROFILE_SAVE_FILE"] = savefolder
    elif savefolder == 'print':
        pass
    elif len(savefolder) > 0:
        os.environ["SIMPA_PROFILE_SAVE_FILE"] = savefolder+"/benchmarking_data_"+profile+"_"+str(spacing)+".txt"

    from simpa_examples.minimal_optical_simulation import run_minimal_optical_simulation
    from simpa_examples.minimal_optical_simulation_uniform_cube import run_minimal_optical_simulation_uniform_cube
    from simpa_examples.optical_and_acoustic_simulation import run_optical_and_acoustic_simulation
    from simpa_examples.segmentation_loader import run_segmentation_loader
    from simpa_examples.three_vs_two_dimensional_simulation_example import run_3Dvs2D_simulation_example

    examples = [run_minimal_optical_simulation,
                run_minimal_optical_simulation_uniform_cube,
                run_optical_and_acoustic_simulation,
                run_segmentation_loader,
                run_3Dvs2D_simulation_example
                ]

    for example in examples:
        example(spacing=spacing, path_manager=None, visualise=False)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run benchmarking tests')
    parser.add_argument("--spacing", default=0.2, help='the voxel spacing in mm')
    parser.add_argument("--profile", default="TIME", type=str,
                        help='the profile to run')
    parser.add_argument("--savefolder", default='default', type=str, help='where to save the results')
    config = parser.parse_args()

    run_benchmarking_tests(spacing=float(config.spacing), profile=config.profile, savefolder=config.savefolder)
