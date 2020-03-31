from ippai.utils import Tags

class SaveFilePaths:
    """
    The save file paths specify the path of a specific data structure in the dictionary of the ippai output hdf5.
    All of these paths have to be used like:
    SaveFilePaths.PATH.format(Tags.UPSAMPLED_DATA or Tags.ORIGINAL_DATA, wavelength)
    """
    SIMULATION_PROPERTIES = "/" + Tags.SIMULATIONS + "/{}/" + Tags.SIMULATION_PROPERTIES + "/{}/"
    OPTICAL_OUTPUT = "/" + Tags.SIMULATIONS + "/{}/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/{}/"
    ACOUSTIC_OUTPUT = "/" + Tags.SIMULATIONS + "/{}/" + Tags.TIME_SERIES_DATA + "/{}/"
    NOISE_ACOUSTIC_OUTPUT = "/" + Tags.SIMULATIONS + "/{}/" + Tags.TIME_SERIES_DATA_NOISE + "/{}/"
    RECONSTRCTION_OUTPUT = "/" + Tags.SIMULATIONS + "/{}/" + Tags.RECONSTRUCTED_DATA + "/{}/"
    NOISE_RECONSTRCTION_OUTPUT = "/" + Tags.SIMULATIONS + "/{}/" + Tags.RECONSTRUCTED_DATA_NOISE + "/{}/"


class SegmentationClasses:
    """
    The segmentation classes define which "tissue types" are modelled in the simulation volumes.
    """
    GENERIC = -1
    AIR = 0
    MUSCLE = 1
    BONE = 2
    BLOOD = 3
    EPIDERMIS = 4
    DERMIS = 5
    FAT = 6
    ULTRASOUND_GEL_PAD = 7
    WATER = 8