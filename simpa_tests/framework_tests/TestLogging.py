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
from simpa.log import Logger
import logging
import os


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.path = "logfile.log"

    def tearDown(self):
        logging.shutdown()
        if os.path.exists(self.path):
            os.remove(self.path)

    def testTwoInstancesAreTheSame(self):
        logger1 = Logger(self.path)
        logger2 = Logger(self.path)
        logger3 = Logger(self.path, force_new_instance=True)

        self.assertEqual(logger1, logger2)
        self.assertNotEqual(logger1, logger3)

    def testLoggingToFile(self):
        logger = Logger(self.path, force_new_instance=True)

        logger.debug("Test")
        logger.info("Test")
        logger.warning("Test")
        logger.error("Test")
        logger.critical("Test")

        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.stat(self.path).st_size > 0)
