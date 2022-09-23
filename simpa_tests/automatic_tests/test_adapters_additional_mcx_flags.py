import unittest
import numpy as np

from simpa import MCXAdapterReflectance, MCXAdapter, Tags, Settings

class TestAdditionalFlagsMCXReflectanceAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.additional_flags = ('-l', '-a')
        general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.MCX_ADDITIONAL_FLAGS: self.additional_flags,
        }
        self.settings = Settings(general_settings)
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_BINARY_PATH: '.',
        })
        self.mcx_reflectance_adapter = MCXAdapterReflectance(global_settings=self.settings)
        self.mcx_adapter = MCXAdapter(global_settings=self.settings)


    def test_get_cmd_mcx_reflectance_adapter(self):
        cmd = self.mcx_reflectance_adapter.get_command()
        self.assertTrue(np.all([f in cmd for f in self.additional_flags]), "Not all additional mcx flags in command returned by mcx reflectance adapter")

    def test_get_cmd_mcx_adapter(self):
        cmd = self.mcx_adapter.get_command()
        self.assertTrue(np.all([f in cmd for f in self.additional_flags]), "Not all additional mcx flags in command returned by mcx reflectance adapter")


if __name__ == '__main__':
    unittest.main()
