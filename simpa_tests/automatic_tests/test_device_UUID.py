# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.core.device_digital_twins import RSOMExplorerP50
from simpa.core.device_digital_twins import InVision256TF
from simpa.core.device_digital_twins import MSOTAcuityEcho
from simpa.core.device_digital_twins import PhotoacousticDevice, LinearArrayDetectionGeometry, \
    PencilArrayIlluminationGeometry


class TestDeviceUUID(unittest.TestCase):

    def testUUIDGeneration(self):
        device1 = RSOMExplorerP50()
        device2 = InVision256TF()
        device3 = MSOTAcuityEcho()
        device4 = PhotoacousticDevice()
        device4.set_detection_geometry(LinearArrayDetectionGeometry())
        device4.add_illumination_geometry(PencilArrayIlluminationGeometry())
        print(device1.generate_uuid())
        print(device2.generate_uuid())
        print(device3.generate_uuid())
        print(device4.generate_uuid())
        print(device1.generate_uuid())
        print(device2.generate_uuid())
        print(device3.generate_uuid())
        print(device4.generate_uuid())
        self.assertEqual(device1.generate_uuid(), device1.generate_uuid())
        self.assertEqual(device2.generate_uuid(), device2.generate_uuid())
        self.assertEqual(device3.generate_uuid(), device3.generate_uuid())
        self.assertEqual(device4.generate_uuid(), device4.generate_uuid())
        self.assertNotEqual(device1.generate_uuid(), device2.generate_uuid())
        self.assertNotEqual(device1.generate_uuid(), device3.generate_uuid())
        self.assertNotEqual(device1.generate_uuid(), device4.generate_uuid())
        self.assertNotEqual(device2.generate_uuid(), device3.generate_uuid())
        self.assertNotEqual(device2.generate_uuid(), device4.generate_uuid())
        self.assertNotEqual(device3.generate_uuid(), device4.generate_uuid())

