"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import unittest
from coverage import Coverage

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cov = Coverage(source=['simpa'])
cov.start()

test_classes = ["simpa_tests.framework_tests.structure_tests.TestLayers",
                "simpa_tests.framework_tests.structure_tests.TestBoxes",
                "simpa_tests.framework_tests.structure_tests.TestEllipticalTubes",
                "simpa_tests.framework_tests.structure_tests.TestParallelEpipeds",
                "simpa_tests.framework_tests.structure_tests.TestSpheres",
                "simpa_tests.framework_tests.structure_tests.TestTubes",
                "simpa_tests.framework_tests.TestPipeline",
                "simpa_tests.framework_tests.TestProcessing",
                "simpa_tests.framework_tests.TestCreateAVolume",
                "simpa_tests.framework_tests.TestIOHandling",
                "simpa_tests.framework_tests.TestCalculationUtils",
                "simpa_tests.framework_tests.TestLogging",
                "simpa_tests.framework_tests.TestPathManager",
                "simpa_tests.framework_tests.tissue_library.TestCoreAssumptions",
                "simpa_tests.framework_tests.tissue_library.TestTissueLibraryAgainstLiteratureValues"
                ]

suite = unittest.TestSuite()
for test_class in test_classes:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(test_class))
unittest.TextTestRunner().run(suite)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../simpa_documentation/test_coverage")
