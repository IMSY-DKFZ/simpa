# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import pathlib
import os

import typer
app = typer.Typer()


@app.command()
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
        savefolder = (str(pathlib.Path(__file__).parent.resolve()) + "/benchmarking_data_" + profile + "_" + str(spacing)
                      + ".txt")
        os.environ["SIMPA_PROFILE_SAVE_FILE"] = savefolder
    elif savefolder == 'print':
        pass
    elif len(savefolder) > 0:
        os.environ["SIMPA_PROFILE_SAVE_FILE"] = savefolder+"/benchmarking_data_"+profile+"_"+str(spacing)+".txt"

    import simpa_examples
    import simpa as sp

    # examples = [simpa_examples.run_linear_unmixing, simpa_examples.run_minimal_optical_simulation,
    #             simpa_examples.run_minimal_optical_simulation_uniform_cube, simpa_examples.run_msot_invision_simulation,
    #             simpa_examples.run_optical_and_acoustic_simulation,
    #             simpa_examples.run_perform_iterative_qPAI_reconstruction]

    examples = [simpa_examples.run_msot_invision_simulation]

    for example in examples:
        try:
            example(SPACING=spacing, path_manager=None, visualise=False)
        except AttributeError:
            print("simulation cannot be run on {} with spacing {}".format(example, spacing))


if __name__ == "__main__":
    app()
