# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from coverage import Coverage
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cov = Coverage(source=['simpa'])
cov.start()

# Discover all tests in the 'simpa_tests' package
loader = unittest.TestLoader()
tests = loader.discover('automatic_tests')  # Specify the directory where your tests are located
print(tests)
runner = unittest.TextTestRunner()
result = runner.run(tests)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../docs/test_coverage")

# Exit with an appropriate code based on the test results
sys.exit(not result.wasSuccessful())