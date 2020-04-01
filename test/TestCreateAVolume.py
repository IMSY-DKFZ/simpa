import unittest
from ippai.utils import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate import SegmentationClasses
from ippai.utils import TISSUE_LIBRARY
import os

class TestTissueProperties(unittest.TestCase):

    def create_background(self):
        water_dict = dict()
        water_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
        water_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.constant(mua=1e-5, mus=1e-5, g=1.0)
        water_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.ULTRASOUND_GEL_PAD
        return water_dict

    def create_vessel(self):
        vessel_dict = dict()
        vessel_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
        vessel_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 12
        vessel_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 12
        vessel_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = 9.5
        vessel_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = 9.5
        vessel_dict[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] = 12
        vessel_dict[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] = 12
        vessel_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = TISSUE_LIBRARY.blood_generic()
        vessel_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
        return vessel_dict

    def create_test_parameters(self):
        structures_dict = dict()
        structures_dict["background"] = self.create_background()
        structures_dict["vessel"] = self.create_vessel()
        return structures_dict

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_create_volume(self):

        random_seed = 4711
        settings = {
            Tags.WAVELENGTHS: [800, 801],
            Tags.RANDOM_SEED: random_seed,
            Tags.VOLUME_NAME: "FlowPhantom_" + str(random_seed).zfill(6),
            Tags.SIMULATION_PATH: ".",
            Tags.RUN_OPTICAL_MODEL: False,
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: False,
            Tags.SPACING_MM: 0.3,
            Tags.DIM_VOLUME_Z_MM: 25,
            Tags.DIM_VOLUME_X_MM: 25,
            Tags.DIM_VOLUME_Y_MM: 10,
            Tags.AIR_LAYER_HEIGHT_MM: 1,
            Tags.GELPAD_LAYER_HEIGHT_MM: 1,
            Tags.STRUCTURES: self.create_test_parameters()
        }
        print("Simulating ", random_seed)
        output = simulate(settings)

        if (os.path.exists(settings[Tags.IPPAI_OUTPUT_PATH]) and
           os.path.isfile(settings[Tags.IPPAI_OUTPUT_PATH])):
            # Delete the created file
            os.remove(settings[Tags.IPPAI_OUTPUT_PATH])
            path = ""
            for subpath in settings[Tags.IPPAI_OUTPUT_PATH].split("/")[:-1]:
                path += subpath + "/"
            # Delete the file's parent directory
            os.rmdir(path)

        for item in output:
            print(item)
        print("Simulating ", random_seed, "[Done]")
