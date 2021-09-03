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

test_classes = ["simpa_tests.automatic_tests.structure_tests.TestLayers",
                "simpa_tests.automatic_tests.structure_tests.TestBoxes",
                "simpa_tests.automatic_tests.structure_tests.TestEllipticalTubes",
                "simpa_tests.automatic_tests.structure_tests.TestParallelEpipeds",
                "simpa_tests.automatic_tests.structure_tests.TestSpheres",
                "simpa_tests.automatic_tests.structure_tests.TestTubes",
                "simpa_tests.automatic_tests.TestPipeline",
                "simpa_tests.automatic_tests.TestProcessing",
                "simpa_tests.automatic_tests.TestCreateAVolume",
                "simpa_tests.automatic_tests.TestIOHandling",
                "simpa_tests.automatic_tests.TestCalculationUtils",
                "simpa_tests.automatic_tests.TestLogging",
                "simpa_tests.automatic_tests.TestPathManager",
                "simpa_tests.automatic_tests.tissue_library.TestCoreAssumptions",
                "simpa_tests.automatic_tests.tissue_library.TestTissueLibraryAgainstLiteratureValues",
                "simpa_tests.automatic_tests.TestNoiseModels",
                "simpa_tests.automatic_tests.TestLinearUnmixing",
                "simpa_tests.automatic_tests.TestIPASCExport",
                "simpa_tests.automatic_tests.TestDeviceUUID",
                ]

suite = unittest.TestSuite()
for test_class in test_classes:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(test_class))
unittest.TextTestRunner().run(suite)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../docs/test_coverage")
