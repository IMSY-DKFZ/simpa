# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import inspect
import os
import unittest
from unittest.mock import patch
from pathlib import Path

from simpa.utils import PathManager
from simpa import Tags


class TestPathManager(unittest.TestCase):
    def setUp(self):
        self.path = '/path_config.env'
        self.save_path = "/workplace/data/"
        self.mcx_path = "/workplace/mcx.exe"
        self.matlab_path = "/workplace/matlab.exe"
        self.file_content = (f"# Example path_config file. Please define all required paths for your simulation here.\n"
                             f"# Afterwards, either copy this file to your current working directory, to your home directory,\n"
                             f"# or to the SIMPA base directry.\n"
                             f"SIMPA_SAVE_PATH={self.save_path}\n"
                             f"MCX_BINARY_PATH={self.mcx_path}\n"
                             f"MATLAB_BINARY_PATH={self.matlab_path}")
        self.home_file = str(Path.home()) + self.path
        self.home_file_exists = os.path.exists(self.home_file)
        self.cwd_file = os.getcwd() + "/" + self.path
        self.cwd_file_exists = os.path.exists(self.cwd_file)
        self.current_file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.simpa_home = self.current_file_path + "/../../" + self.path
        self.simpa_home_exists = os.path.exists(self.simpa_home)
  
    @unittest.expectedFailure
    def test_variables_not_set():
        path_manager = PathManager()
        _ = path_manager.get_mcx_binary_path()
        _ = path_manager.get_hdf5_file_save_path()
        _ = path_manager.get_matlab_binary_path()

    @patch.dict(os.environ, {Tags.SIMPA_SAVE_PATH_VARNAME: "test_simpa_save_path",
                             Tags.MCX_BINARY_PATH_VARNAME: "test_mcx_path"})
    def test_instantiate_without_file(self):
        path_manager = PathManager()
        self.assertEqual(path_manager.get_mcx_binary_path(), "test_mcx_path")
        self.assertEqual(path_manager.get_hdf5_file_save_path(), "test_simpa_save_path")

    def test_instantiate_when_file_is_in_home(self):

        if self.home_file_exists:
            self.hide_config_file(self.home_file)
        self.write_config_file(self.home_file)

        path_manager = PathManager()
        self.check_path_manager_correctly_loaded(path_manager)

        self.delete_config_file(self.home_file)
        if self.home_file_exists:
            self.restore_config_file(self.home_file)

    def test_instantiate_when_file_is_in_cwd(self):
        if self.home_file_exists:
            self.hide_config_file(self.home_file)
        if self.simpa_home_exists:
            self.hide_config_file(self.simpa_home)
        if not self.cwd_file_exists:
            self.write_config_file(self.cwd_file)

        path_manager = PathManager()
        self.check_path_manager_correctly_loaded(path_manager)

        if self.home_file_exists:
            self.restore_config_file(self.home_file)
        if self.simpa_home_exists:
            self.restore_config_file(self.simpa_home)
        if not self.cwd_file_exists:
            self.delete_config_file(self.cwd_file)

    def test_instantiate_when_file_is_in_simpa_home(self):
        if self.home_file_exists:
            self.hide_config_file(self.home_file)
        if self.cwd_file_exists:
            self.hide_config_file(self.cwd_file)
        if not self.simpa_home_exists:
            self.write_config_file(self.simpa_home)

        path_manager = PathManager()
        self.check_path_manager_correctly_loaded(path_manager)

        if self.home_file_exists:
            self.restore_config_file(self.home_file)
        if self.cwd_file_exists:
            self.restore_config_file(self.cwd_file)
        if not self.simpa_home_exists:
            self.delete_config_file(self.simpa_home)

    def check_path_manager_correctly_loaded(self, path_manager: PathManager):
        self.assertEqual(path_manager.get_hdf5_file_save_path(), self.save_path)
        self.assertEqual(path_manager.get_mcx_binary_path(), self.mcx_path)
        self.assertEqual(path_manager.get_matlab_binary_path(), self.matlab_path)

    def write_config_file(self, path):
        with open(path, "w") as write_path:
            write_path.writelines(self.file_content)

    def delete_config_file(self, path):
        os.remove(path)

    def hide_config_file(self, path: str):
        os.rename(path, path.replace("path_config.env", "path_config.env.backup"))

    def restore_config_file(self, path: str):
        os.rename(path.replace("path_config.env", "path_config.env.backup"), path)
