# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import glob
import os
import inspect

base_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

files = glob.glob(os.path.join(base_path, "automatic_tests", "*/*.py"), recursive=True)
files += glob.glob(os.path.join(base_path, "automatic_tests", "*.py"), recursive=True)
files += glob.glob(os.path.join(base_path, "automatic_tests", "*/*/*.py"), recursive=True)

automatic_test_classes = [file.replace(os.path.sep, ".")[file.replace(os.path.sep, ".").find("simpa_tests"):-3]
                          for file in files]
