# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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

from ippai.utils import Tags
from ippai.simulate import SaveFilePaths
from ippai.simulate.models.optical_models.mcx_adapter import McxAdapter
from ippai.simulate.models.optical_models.mcxyz_adapter import McxyzAdapter
from ippai.io_handling.io_hdf5 import save_hdf5, load_hdf5


def run_optical_forward_model(settings, optical_properties_path):
    # TODO
    print("OPTICAL FORWARD")

    if Tags.OPTICAL_MODEL not in settings:
        raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings. Skipping optical modelling.")

    model = settings[Tags.OPTICAL_MODEL]
    forward_model_implementation = None

    if model == Tags.MODEL_MCXYZ:
        forward_model_implementation = McxyzAdapter()
    elif model == Tags.MODEL_MCX:
        forward_model_implementation = McxAdapter()
    # elif model == Tags.MODEL_TEST_OPTICAL: TODO
    #     forward_model_implementation =


    fluence = forward_model_implementation.simulate(optical_properties_path, settings)

    optical_properties = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], optical_properties_path)
    absorption = optical_properties[Tags.PROPERTY_ABSORPTION_PER_CM]
    gruneisen_parameter = optical_properties[Tags.PROPERTY_GRUNEISEN_PARAMETER]

    if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
        units = Tags.UNITS_PRESSURE
        # Initial pressure should be given in units of Pascale
        conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
        initial_pressure = (absorption * fluence * gruneisen_parameter *
                            (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000)
                            * conversion_factor)
    else:
        units = Tags.UNITS_ARBITRARY
        initial_pressure = absorption * fluence

    optical_output_path = SaveFilePaths.OPTICAL_OUTPUT.\
        format(Tags.ORIGINAL_DATA, settings[Tags.WAVELENGTH])

    save_hdf5({Tags.OPTICAL_MODEL_FLUENCE: fluence,
               Tags.OPTICAL_MODEL_INITIAL_PRESSURE: initial_pressure,
               Tags.OPTICAL_MODEL_UNITS: units},
              settings[Tags.IPPAI_OUTPUT_PATH],
              optical_output_path)

    return optical_output_path
