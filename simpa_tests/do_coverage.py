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
from coverage import Coverage

cov = Coverage(source=['simpa'])
cov.start()

test_classes = ["simpa_tests.framework_tests.structure_tests.TestLayers",
                "simpa_tests.framework_tests.structure_tests.TestBoxes",
                "simpa_tests.framework_tests.structure_tests.TestEllipticalTubes",
                "simpa_tests.framework_tests.structure_tests.TestParallelEpipeds",
                "simpa_tests.framework_tests.structure_tests.TestSpheres",
                "simpa_tests.framework_tests.structure_tests.TestTubes",
                "simpa_tests.framework_tests.TestPipeline",
                "simpa_tests.framework_tests.TestCreateAVolume",
                "simpa_tests.framework_tests.TestCreateSettings",
                "simpa_tests.framework_tests.TestIOHandling",
                "simpa_tests.framework_tests.TestCalculationUtils",
                "simpa_tests.framework_tests.TestMoleculeLibrary"
                ]

suite = unittest.TestSuite()
for test_class in test_classes:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(test_class))
unittest.TextTestRunner().run(suite)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../simpa_documentation/test_coverage")
