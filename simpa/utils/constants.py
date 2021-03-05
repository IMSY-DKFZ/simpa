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


class SaveFilePaths:
    """
    The save file paths specify the path of a specific data structure in the dictionary of the simpa output hdf5.
    All of these paths have to be used like:
    SaveFilePaths.PATH + "data_structure"
    """
    SIMULATION_PROPERTIES = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/"
    OPTICAL_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/"
    ACOUSTIC_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.TIME_SERIES_DATA + "/"
    NOISE_ACOUSTIC_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.TIME_SERIES_DATA_NOISE + "/"
    RECONSTRCTION_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.RECONSTRUCTED_DATA + "/"
    NOISE_RECONSTRCTION_OUTPUT = "/" + Tags.SIMULATIONS + "/" + Tags.RECONSTRUCTED_DATA_NOISE + "/"


class SegmentationClasses:
    """
    The segmentation classes define which "tissue types" are modelled in the simulation volumes.
    """
    GENERIC = -1
    AIR = 0
    MUSCLE = 1
    BONE = 2
    BLOOD = 3
    EPIDERMIS = 4
    DERMIS = 5
    FAT = 6
    ULTRASOUND_GEL = 7
    WATER = 8
    HEAVY_WATER = 9
    COUPLING_ARTIFACT = 10
    MEDIPRENE = 11
