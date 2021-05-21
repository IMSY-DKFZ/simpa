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

from simpa.utils import Tags, SegmentationClasses
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.tissue_properties import TissueProperties
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def validate_expected_values_dictionary(expected_values: dict):

    if len(expected_values.keys()) < 1:
        raise ValueError("The expected_values dictionary must have at least one entry.")

    for key in expected_values.keys():
        if not isinstance(key, (float, int)):
            raise KeyError("The keys of the expected_values dictionary must be numbers.")

        if not isinstance(expected_values[key], TissueProperties):
            raise ValueError("The expected_values dictionary values must be TissueProperty instances.")


def compare_molecular_composition_against_expected_values(molecular_composition: MolecularComposition,
                                                          expected_values: dict,
                                                          tolerated_margin_in_percent: float = 0.05,
                                                          visualise_values: bool = False):
    """
    :param molecular_composition:
    :param expected_values:
    :param tolerated_margin_in_percent:
    """

    validate_expected_values_dictionary(expected_values)
    if visualise_values:
        plt.figure()
        num_subplots = len(TissueProperties.property_tags)

    for wavelength in expected_values.keys():
        molecular_composition.update_internal_properties()
        composition_properties = molecular_composition.get_properties_for_wavelength(wavelength=wavelength)
        expected_properties = expected_values[wavelength]

        if visualise_values:
            for tag_idx, tag in enumerate(TissueProperties.property_tags):
                ax = plt.subplot(3, np.ceil(num_subplots/3), (tag_idx+1))
                ax.set_title(tag)
                ax.set_xlabel("wavelength [nm]")
                ax.set_ylabel("value [units]")
                ax.scatter(x=wavelength, y=composition_properties[tag], c="blue")
                ax.add_patch(patches.Rectangle((wavelength-10, expected_properties[tag] * (1-tolerated_margin_in_percent)),
                                               20, expected_properties[tag] * 2 * tolerated_margin_in_percent,
                                               color="green", alpha=0.2))
                ax.scatter(x=wavelength, y=expected_properties[tag], c="green")
        else:
            for tag in TissueProperties.property_tags:
                if (np.abs(composition_properties[tag] - expected_properties[tag]) /
                        expected_properties[tag] > tolerated_margin_in_percent):
                    raise AssertionError(f"The calculated value for {tag} at "
                                         f"wavelength {wavelength}nm was different from the"
                                         f" expected value by a margin greater than {tolerated_margin_in_percent*100}%"
                                         f" (was {composition_properties[tag]} but was "
                                         f"expected to be {expected_properties[tag]})")

    if visualise_values:
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    values700nm = TissueProperties()
    values700nm[Tags.PROPERTY_ABSORPTION_PER_CM] = 10.0
    values700nm[Tags.PROPERTY_SCATTERING_PER_CM] = 200.0
    values700nm[Tags.PROPERTY_ANISOTROPY] = 0.87
    values700nm[Tags.PROPERTY_GRUNEISEN_PARAMETER] = 0.15
    values700nm[Tags.PROPERTY_SEGMENTATION] = SegmentationClasses.BLOOD
    values700nm[Tags.PROPERTY_OXYGENATION] = 0.0
    values700nm[Tags.PROPERTY_DENSITY] = 1500
    values700nm[Tags.PROPERTY_SPEED_OF_SOUND] = 1500
    values700nm[Tags.PROPERTY_ALPHA_COEFF] = 1.1

    values800nm = TissueProperties()
    values800nm[Tags.PROPERTY_ABSORPTION_PER_CM] = 4
    values800nm[Tags.PROPERTY_SCATTERING_PER_CM] = 200.0
    values800nm[Tags.PROPERTY_ANISOTROPY] = 0.87
    values800nm[Tags.PROPERTY_GRUNEISEN_PARAMETER] = 0.15
    values800nm[Tags.PROPERTY_SEGMENTATION] = SegmentationClasses.BLOOD
    values800nm[Tags.PROPERTY_OXYGENATION] = 0.0
    values800nm[Tags.PROPERTY_DENSITY] = 1500
    values800nm[Tags.PROPERTY_SPEED_OF_SOUND] = 1500
    values800nm[Tags.PROPERTY_ALPHA_COEFF] = 1.1

    values900nm = TissueProperties()
    values900nm[Tags.PROPERTY_ABSORPTION_PER_CM] = 5
    values900nm[Tags.PROPERTY_SCATTERING_PER_CM] = 200.0
    values900nm[Tags.PROPERTY_ANISOTROPY] = 0.87
    values900nm[Tags.PROPERTY_GRUNEISEN_PARAMETER] = 0.15
    values900nm[Tags.PROPERTY_SEGMENTATION] = SegmentationClasses.BLOOD
    values900nm[Tags.PROPERTY_OXYGENATION] = 0.0
    values900nm[Tags.PROPERTY_DENSITY] = 1500
    values900nm[Tags.PROPERTY_SPEED_OF_SOUND] = 1500
    values900nm[Tags.PROPERTY_ALPHA_COEFF] = 1.1

    expectd_values = {
        700: values700nm,
        800: values800nm,
        900: values900nm
    }

    compare_molecular_composition_against_expected_values(molecular_composition=TISSUE_LIBRARY.dermis(0.0),
                                                          expected_values=expectd_values,
                                                          visualise_values=True)
