import unittest
import numpy as np

from simpa import MCXAdapterReflectance, Tags, PathManager, Settings

path_manager = PathManager()


class TestMCXReflectanceAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.additional_flags = ('-l', '-a')
        general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: 123,
            Tags.VOLUME_NAME: 'test',
            Tags.SPACING_MM: 15,
            Tags.DIM_VOLUME_Z_MM: 13,
            Tags.DIM_VOLUME_X_MM: 13,
            Tags.DIM_VOLUME_Y_MM: 13,
            Tags.WAVELENGTHS: [798],
            Tags.DO_FILE_COMPRESSION: True,
            Tags.MCX_ADDITIONAL_FLAGS: self.additional_flags,
        }
        self.settings = Settings(general_settings)
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: '.',
            Tags.COMPUTE_DIFFUSE_REFLECTANCE: False,
            Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT: False
        })
        self.adapter = MCXAdapterReflectance(global_settings=self.settings)

    def test_get_cmd(self):
        cmd = self.adapter.get_command()
        self.assertTrue(np.all([f in cmd for f in self.additional_flags]), "Not all additional flags in command")


if __name__ == '__main__':
    unittest.main()
