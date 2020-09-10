# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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

test_classes = ["test.framework_tests.TestPipeline",
                "test.framework_tests.TestCreateAVolume",
                "test.framework_tests.TestIOHandling",
                "test.framework_tests.TestTissueProperties"]

suite = unittest.TestSuite()
for test_class in test_classes:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(test_class))
unittest.TextTestRunner().run(suite)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../documentation/test_coverage")