# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from coverage import Coverage
from simpa_tests import automatic_test_classes

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cov = Coverage(source=['simpa'])
cov.start()

suite = unittest.TestSuite()
for test_class in automatic_test_classes:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(test_class))
unittest.TextTestRunner().run(suite)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../docs/test_coverage")
