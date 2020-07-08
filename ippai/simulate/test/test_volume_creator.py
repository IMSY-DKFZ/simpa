
from unittest import TestCase
import numpy as np
from typing import Union, Tuple

from ippai.simulate.volume_creator import set_custom_parameter_map
from ippai.utils.tags import Tags
from ippai.simulate.structures import create_custom_2d_map


class TestCustomParameterMap(TestCase):
    def setUp(self, target_shape: Union[None, Tuple] = None, map_shape: Union[None, Tuple] = None) -> None:
        if target_shape is not None:
            self.target_shape = target_shape
        else:
            self.target_shape = (23, 17, 1)
        if map_shape is not None:
            self.map_shape = map_shape
        else:
            self.map_shape = (7, 5)
        self.param_tag = 'vhb'
        self.volumes = {self.param_tag: np.zeros(self.target_shape)}
        self.custom_map = np.random.random(self.map_shape)
        self.global_settings = {Tags.CREATE_AXIS_SYMMETRICAL_VOLUME: True,
                                Tags.VOLUME_SYMMETRY_AXIS: 2}
        self.structure_settings = create_custom_2d_map(self.custom_map, self.param_tag)

    def test_set_custom_parameter_map_2d_upscale(self):
        self.setUp(target_shape=(23, 17, 1), map_shape=(7, 5))
        new_volume = set_custom_parameter_map(self.volumes, self.structure_settings, self.global_settings)
        if new_volume[self.param_tag].shape != self.target_shape:
            self.fail(f"Target shape: {self.target_shape} doe snot match shape after custom map initialization: "
                      f"{new_volume[self.param_tag].shape}")
        if not new_volume[self.param_tag].sum():
            self.fail("Volume did not get updated, all values in volume are zero")

    def test_set_custom_parameter_map_2d_downscale(self):
        self.setUp(target_shape=(7, 11, 1), map_shape=(107, 113))
        new_volume = set_custom_parameter_map(self.volumes, self.structure_settings, self.global_settings)
        if new_volume[self.param_tag].shape != self.target_shape:
            self.fail(f"Target shape: {self.target_shape} doe snot match shape after custom map initialization: "
                      f"{new_volume[self.param_tag].shape}")
        if not new_volume[self.param_tag].sum():
            self.fail("Volume did not get updated, all values in volume are zero")

    def test_set_custom_parameter_map_3d_downscale(self):
        self.setUp(target_shape=(7, 11, 17), map_shape=(107, 113, 31))
        new_volume = set_custom_parameter_map(self.volumes, self.structure_settings, self.global_settings)
        if new_volume[self.param_tag].shape != self.target_shape:
            self.fail(f"Target shape: {self.target_shape} doe snot match shape after custom map initialization: "
                      f"{new_volume[self.param_tag].shape}")
        if not new_volume[self.param_tag].sum():
            self.fail("Volume did not get updated, all values in volume are zero")

    def test_set_custom_parameter_map_3d_upscale(self):
        self.setUp(target_shape=(107, 113, 31), map_shape=(7, 11, 17))
        new_volume = set_custom_parameter_map(self.volumes, self.structure_settings, self.global_settings)
        if new_volume[self.param_tag].shape != self.target_shape:
            self.fail(f"Target shape: {self.target_shape} doe snot match shape after custom map initialization: "
                      f"{new_volume[self.param_tag].shape}")
        if not new_volume[self.param_tag].sum():
            self.fail("Volume did not get updated, all values in volume are zero")