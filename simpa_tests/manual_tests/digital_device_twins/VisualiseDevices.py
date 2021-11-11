# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
import numpy as np
from simpa_tests.manual_tests import ManualIntegrationTestClass


class DeviceVisualisationTest(ManualIntegrationTestClass):

    def setup(self):
        pass

    def perform_test(self):
        pass

    def tear_down(self):
        pass

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        sp.visualise_device(sp.MSOTAcuityEcho(device_position_mm=np.asarray([50, 10, 0])),
                            save_path + "device_visualisation_MSOT_Acuity.png")
        sp.visualise_device(sp.InVision256TF(device_position_mm=np.asarray([50, 10, 50])),
                            save_path + "device_visualisation_MSOT_Invision.png")
        sp.visualise_device(sp.RSOMExplorerP50(device_position_mm=np.asarray([50, 10, 0])),
                            save_path + "device_visualisation_RSOM_Explorer.png")


if __name__ == "__main__":
    test = DeviceVisualisationTest()
    test.run_test(show_figure_on_screen=False)