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

import unittest
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.core.simulation import simulate
import numpy as np
from simpa_tests.test_utils import create_test_structure_parameters
import os
from simpa.core.pipeline_components import ModelBasedVolumeCreator
from simpa.core.optical_simulation.test_optical_adapter import TestOpticalComponent
from simpa.core.acoustic_simulation.test_acoustic_adapter import TestAcousticModelAdapter

class TestPipeline(unittest.TestCase):

    def setUp(self):

        self.VOLUME_WIDTH_IN_MM = 4
        self.VOLUME_HEIGHT_IN_MM = 3
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711

    def test_pipeline(self):
        # Seed the numpy random configuration prior to creating the settings file in
        # order to ensure that the same volume
        # is generated with the same random seed every time.

        np.random.seed(self.RANDOM_SEED)

        settings = {
            # These parameters set the general propeties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: "TestName_"+str(self.RANDOM_SEED),
            Tags.SIMULATION_PATH: ".",
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_WIDTH_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_WIDTH_IN_MM,

            # The following parameters set the optical forward model
            Tags.WAVELENGTHS: [800],


            # The following parameters tell the script that we do not want any extra
            # modelling steps
            # Add the volume_creation to be simulated to the tissue
        }

        settings = Settings(settings)

        settings["optical_settings"] = {
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_TEST,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
        }

        simulation_pipeline = [
            ModelBasedVolumeCreator(settings, "optical_settings"),
            TestOpticalComponent(settings, "optical_settings"),
            TestAcousticModelAdapter(settings, "optical_settings"),
        ]

        settings[Tags.STRUCTURES] = create_test_structure_parameters(settings)
        simulate(simulation_pipeline, settings)

        if (os.path.exists(settings[Tags.SIMPA_OUTPUT_PATH]) and
                os.path.isfile(settings[Tags.SIMPA_OUTPUT_PATH])):
            # Delete the created file
            os.remove(settings[Tags.SIMPA_OUTPUT_PATH])
