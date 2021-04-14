# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from simpa.utils import Tags


def define_illumination(global_settings, optical_simulation_settings, nx, ny, nz):
    """
        This method creates a dictionary that represents the illumination geometry in a way
        that it can be used with the respective illumination framework.

        :param global_settings: The top-level settings dictionary
        :param optical_simulation_settings: The settings file containing the simulation instructions
        :param nx: number of voxels along the x dimension of the volume
        :param ny: number of voxels along the y dimension of the volume
        :param nz: number of voxels along the z dimension of the volume
        """
    return define_illumination_mcx(global_settings, optical_simulation_settings, nx, ny, nz)


def define_illumination_mcx(global_settings, optical_simulation_settings, nx, ny, nz) -> dict:
    """
    This method creates a dictionary that contains tags as they are expected for the
    mcx simulation tool to represent the illumination geometry.

    :param global_settings: The top-level settings dictionary
    :param optical_simulation_settings: The settings file containing the simulation instructions
    :param nx: number of voxels along the x dimension of the volume
    :param ny: number of voxels along the y dimension of the volume
    :param nz: number of voxels along the z dimension of the volume
    """
    if Tags.ILLUMINATION_TYPE not in optical_simulation_settings:
        source_type = Tags.ILLUMINATION_TYPE_PENCIL
    else:
        source_type = optical_simulation_settings[Tags.ILLUMINATION_TYPE]

    if optical_simulation_settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO:
        source_position = [int(nx/2.0) + 0.5, int(ny/2.0 - 17.81/global_settings[Tags.SPACING_MM]) + 0.5, 1]
    elif optical_simulation_settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_INVISION:
        source_position = [int(nx / 2.0) + 0.5, int(ny / 2.0) + 0.5, int(nz / 2.0) + 0.5]
    elif Tags.ILLUMINATION_POSITION not in optical_simulation_settings:
        source_position = [int(nx / 2.0) + 0.5, int(ny / 2.0) + 0.5, 1]
    else:
        source_position = optical_simulation_settings[Tags.ILLUMINATION_POSITION]

    if optical_simulation_settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO:
        source_direction = [0, 0.381070, 0.9245460]
    elif Tags.ILLUMINATION_DIRECTION not in optical_simulation_settings:
        source_direction = [0, 0, 1]
    else:
        source_direction = optical_simulation_settings[Tags.ILLUMINATION_DIRECTION]

    if optical_simulation_settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO:
        source_param1 = [30 / global_settings[Tags.SPACING_MM], 0, 0, 0]
    elif optical_simulation_settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_INVISION:
        source_param1 = [global_settings[Tags.SPACING_MM], 0, 0, 0]
    elif Tags.ILLUMINATION_PARAM1 not in optical_simulation_settings:
        source_param1 = [0, 0, 0, 0]
    else:
        source_param1 = optical_simulation_settings[Tags.ILLUMINATION_PARAM1]

    if Tags.ILLUMINATION_PARAM2 not in optical_simulation_settings:
        source_param2 = [0, 0, 0, 0]
    else:
        source_param2 = optical_simulation_settings[Tags.ILLUMINATION_PARAM2]

    return {
        "Type": source_type,
        "Pos": source_position,
        "Dir": source_direction,
        "Param1": source_param1,
        "Param2": source_param2
    }
