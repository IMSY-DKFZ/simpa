# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

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
