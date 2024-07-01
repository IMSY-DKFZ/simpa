# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def get_final_table(savefolder: str = None):
    """ Function to get the final table from a benchmarking file. Just call it after the last iteration of the
    benchmarking. Saves csv in same location as savefolder!
    :param savefolder:  str to csv file containing benchmarking data frame
    :return: None
    :raise: None
    """
    if savefolder is None or savefolder == "default":
        savefolder = Path(__file__).parent.resolve()
    else:
        savefolder = Path(savefolder)

    # read the csv file
    df_file = savefolder / 'benchmarking_data_frame.csv'
    new_df = pd.read_csv(df_file)

    # init result
    mean_list = []
    # loop over every example, spacing, profile and saves mean and std for the sub dataframe
    for expl in np.unique(new_df['Example']):
        for space in np.unique(new_df['Spacing']):
            expl_list = [expl, space]  # init row of result
            for prof in np.unique(new_df['Profile']):
                # get the sub data frame for a specific profile and spacing.
                sub_df = new_df.loc[
                    (new_df['Spacing'] == space) & (new_df['Example'] == expl) & (new_df['Profile'] == prof)]
                _mean = sub_df.mean(numeric_only=True)['Value']
                _std = sub_df.std(numeric_only=True)['Value']
                expl_list.append(np.round(_mean, 2))  # append to output row
                expl_list.append(np.round(_std, 2))
            mean_list.append(expl_list)

    # init naming and format of output data frame columns
    my_profiles = []
    type_dict = {"Example": "str", "Spacing": "float64"}
    for entry in np.unique(new_df['Profile']).tolist():
        my_profiles.append(entry + '_mean')
        my_profiles.append(entry + '_std')
        type_dict[entry + '_mean'] = 'float64'
        type_dict[entry + '_std'] = 'float64'

    # concat rows (list of lists) to a single data frame
    cols = ['Example', 'Spacing'] + my_profiles
    mean_df = pd.DataFrame(mean_list, columns=cols)
    mean_df.astype(dtype=type_dict)
    # save to csv at input location
    mean_df.to_csv(str(df_file).replace('.csv', '_mean.csv'), index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run benchmarking tests')
    parser.add_argument("--savefolder", default=None, type=str, help='where to save the results')
    config = parser.parse_args()

    savefolder = config.savefolder
    get_final_table(savefolder=savefolder)
