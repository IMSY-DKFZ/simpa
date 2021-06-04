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
from simpa.core.simulation import simulate
from simpa.algorithms.multispectral import linear_unmixing as lu
import numpy as np
from simpa.core import *
from simpa.utils.path_manager import PathManager
from simpa.io_handling import load_data_field, load_hdf5, save_hdf5
from simpa.log import Logger
import os
from simpa.core.device_digital_twins import PencilBeamIlluminationGeometry


class TestLinearUnmixing:

    def setup(self):

        self.logger = Logger()
        # TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
        #  point to the correct file in the PathManager().
        self.path_manager = PathManager()
        RANDOM_SEED = 471
        self.WAVELENGTHS = [750, 850]

        general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "LinearUnmixingAutomaticTest_" + str(RANDOM_SEED),
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: 5,
            Tags.DIM_VOLUME_Z_MM: 25,
            Tags.DIM_VOLUME_X_MM: 5,
            Tags.DIM_VOLUME_Y_MM: 5,
            Tags.WAVELENGTHS: self.WAVELENGTHS
        }
        self.settings = Settings(general_settings)

        self.device = PencilBeamIlluminationGeometry()

        pipeline = []
        simulate(pipeline, self.settings, self.device)

        oxy = np.array([2.77, 5.67])
        deoxy = np.array([7.52, 3.7])

        test_750_mua = np.array([[[0., oxy[0] + deoxy[0], oxy[0], deoxy[0], 0.7 * oxy[0] + 0.3 * deoxy[0]]]])
        test_850_mua = np.array([[[0., oxy[1] + deoxy[1], oxy[1], deoxy[1], 0.7 * oxy[1] + 0.3 * deoxy[1]]]])

        self.file = load_hdf5(self.settings[Tags.SIMPA_OUTPUT_PATH])["settings"]

        mua_dict = {
            "750": test_750_mua,
            "850": test_850_mua
        }

        self.file["simulations"] = {
            "simulation_properties": {"mua": mua_dict}
            }

    def perform_test(self):

        self.logger.info("Performing generic linear unmixing test...")
        save_hdf5(self.file, self.settings[Tags.SIMPA_OUTPUT_PATH])

        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.PROPERTY_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_OXYHEMOGLOBIN: self.WAVELENGTHS,
            Tags.LINEAR_UNMIXING_DEOXYHEMOGLOBIN: self.WAVELENGTHS,
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True
        }

        lu.LinearUnmixingProcessingComponent(self.settings, "linear_unmixing").run(self.device)

        lu_results = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        sO2 = lu_results["sO2"]

        try:
            assert np.allclose(sO2, np.array([[[0, 0.5, 1, 0, 0.7]]]), atol=1e-8)
            self.logger.info("Linear unmixing test successful!")
        except:
            self.logger.critical("Linear unmixing test failed!")

        # clean up files after test
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

if __name__ == '__main__':
    test = TestLinearUnmixing()
    test.setup()
    test.perform_test()

