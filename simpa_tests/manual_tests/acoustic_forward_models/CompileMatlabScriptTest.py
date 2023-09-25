# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import os
import shutil

from simpa.utils.path_manager import PathManager
from simpa.utils.matlab import generate_compiled_matlab_scripts
from simpa_tests.manual_tests import ManualIntegrationTestClass


kwave_binary_path = ''  # specify before running the test

class CompileMatlabScriptTest(ManualIntegrationTestClass):

    def setup(self):
        assert kwave_binary_path, 'kwave_binary_path must be specified to run this test'

        path_manager = PathManager()
        self.matlab_binary_path = path_manager.get_matlab_binary_path()
        self.kwave_binary_path = kwave_binary_path
        self.matlab_compiled_script_path = path_manager.get_matlab_compiled_script_path()
        self.hidden_name = ''
        if os.path.exists(self.matlab_compiled_script_path):
            self.hidden_name = self.matlab_compiled_script_path + ".backup"
            if os.path.exists(self.hidden_name):
                shutil.rmtree(self.hidden_name)
            os.rename(self.matlab_compiled_script_path, self.hidden_name)


    def perform_test(self):
        generate_compiled_matlab_scripts(self.matlab_binary_path, self.kwave_binary_path, self.matlab_compiled_script_path)
        assert os.path.exists(os.path.join(self.matlab_compiled_script_path, "simulate_2D", "run_simulate_2D.sh"))
        assert os.path.exists(os.path.join(self.matlab_compiled_script_path, "simulate_3D", "run_simulate_3D.sh"))
        assert os.path.exists(os.path.join(self.matlab_compiled_script_path, "time_reversal_2D", "run_time_reversal_2D.sh"))
        assert os.path.exists(os.path.join(self.matlab_compiled_script_path, "time_reversal_3D", "run_time_reversal_3D.sh"))

    def tear_down(self):
        shutil.rmtree(self.matlab_compiled_script_path, ignore_errors=True)
        if self.hidden_name:
            os.rename(self.hidden_name, self.matlab_compiled_script_path)

if __name__ == "__main__":
    test = CompileMatlabScriptTest()
    test.run_test()