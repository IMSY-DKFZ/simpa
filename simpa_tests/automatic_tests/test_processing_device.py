# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.log.file_logger import Logger
from simpa.utils.tags import Tags
from simpa.utils.settings import Settings
from simpa.utils.processing_device import get_processing_device
import unittest
from unittest.mock import Mock
import torch


class TestProcessing(unittest.TestCase):

    def setUp(self):
        print("setUp")

        self.settings_without_tag = Settings()

        self.settings_with_GPU = Settings({
            Tags.GPU: True
        })

        self.settings_with_CPU = Settings({
            Tags.GPU: False
        })

    def test_get_processing_device_with_no_settings(self):
        '''
        If cuda is available and no settings are given by the user, verify that a torch GPU device is returned.
        '''

        print("test get_processing_device with no settings")

        # mock pre-requisite for this test case that cuda is available
        temp_cuda = torch.cuda
        mock = Mock()
        mock.is_available.return_value = True
        torch.cuda = mock
        assert torch.cuda.is_available(), "Check that CUDA is avilable, otherwise the CPU will always be used"

        # actual test case
        device = get_processing_device(global_settings=None)
        assert device == torch.device("cuda"), f"Processing device is not the assumed torch GPU device, but {device}"

        # mock pre-requisite for this test case that cuda is not available
        torch.cuda.is_available.return_value = False
        assert not torch.cuda.is_available(), "Check that CUDA is not avilable"

        # actual test case
        device = get_processing_device(global_settings=None)
        assert device == torch.device("cpu"), f"Processing device is not the assumed torch CPU device, but {device}"

        # restore torch.cuda
        torch.cuda = temp_cuda

    def test_get_processing_device_with_settings_without_tag(self):
        '''
        If cuda is available and no Tag is specified in settings by user, verify that a torch GPU device is returned
        '''

        print("test get_processing_device with settings without tag")

        # mock pre-requisite for this test case that cuda is available
        temp_cuda = torch.cuda
        mock = Mock()
        mock.is_available.return_value = True
        torch.cuda = mock
        assert torch.cuda.is_available(), "Check that CUDA is avilable, otherwise the CPU will always be used"

        # actual test case
        device = get_processing_device(global_settings=self.settings_without_tag)
        assert device == torch.device("cuda"), f"Processing device is not the assumed torch GPU device, but {device}"

        # mock pre-requisite for this test case that cuda is not available
        torch.cuda.is_available.return_value = False
        assert not torch.cuda.is_available(), "Check that CUDA is not avilable"

        # actual test case
        device = get_processing_device(global_settings=self.settings_without_tag)
        assert device == torch.device("cpu"), f"Processing device is not the assumed torch CPU device, but {device}"

        # restore torch.cuda
        torch.cuda = temp_cuda

    def test_get_processing_device_with_settings_with_gpu(self):
        '''
        If cuda is available and GPU Tag is specified as True in settings by user, verify that a torch GPU device is returned
        '''

        print("test get_processing_device with settings with GPU tag")

        # mock pre-requisite for this test case that cuda is available
        temp_cuda = torch.cuda
        mock = Mock()
        mock.is_available.return_value = True
        torch.cuda = mock
        assert torch.cuda.is_available(), "Check that CUDA is avilable, otherwise the CPU will always be used"

        # actual test case
        device = get_processing_device(global_settings=self.settings_with_GPU)
        assert device == torch.device("cuda"), f"Processing device is not the assumed torch GPU device, but {device}"

        # mock pre-requisite for this test case that cuda is not available
        torch.cuda.is_available.return_value = False
        assert not torch.cuda.is_available(), "Check that CUDA is not avilable"

        # actual test case
        with self.assertLogs("SIMPA Logger", level='WARN') as context_manager:
            device = get_processing_device(global_settings=self.settings_with_GPU)

            assert device == torch.device("cpu"), f"Processing device is not the assumed torch CPU device, but {device}"

            # also check if warning is logged
            assert context_manager.output == [
                'WARNING:SIMPA Logger:Cuda is not available! Check your torch/cuda version. Processing will be done on CPU instead.'], "Warning that CPU instead of GPU is used is not logged"

        # restore torch.cuda
        torch.cuda = temp_cuda

    def test_get_processing_device_with_settings_with_cpu(self):
        '''
        If cuda is available or not and GPU Tag is specified as False in settings by user, verify that a torch CPU device is returned
        '''

        print("test get_processing_device with settings with GPU tag set to False")

        # mock pre-requisite for this test case that cuda is available
        temp_cuda = torch.cuda
        mock = Mock()
        mock.is_available.return_value = True
        torch.cuda = mock
        assert torch.cuda.is_available(), "Check that CUDA is avilable, otherwise the CPU will always be used"

        # actual test case
        device = get_processing_device(global_settings=self.settings_with_CPU)
        assert device == torch.device("cpu"), f"Processing device is not the assumed torch CPU device, but {device}"

        # mock pre-requisite for this test case that cuda is not available
        torch.cuda.is_available.return_value = False
        assert not torch.cuda.is_available(), "Check that CUDA is not avilable"

        # actual test case
        device = get_processing_device(global_settings=self.settings_with_CPU)
        assert device == torch.device("cpu"), f"Processing device is not the assumed torch CPU device, but {device}"

        # restore torch.cuda
        torch.cuda = temp_cuda


if __name__ == "__main__":
    test = TestProcessing()
    test.setUp()
    test.test_get_processing_device_with_no_settings()
    test.test_get_processing_device_with_settings_without_tag()
    test.test_get_processing_device_with_settings_with_gpu()
    test.test_get_processing_device_with_settings_with_cpu()
