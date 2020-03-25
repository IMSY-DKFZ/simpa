from json import JSONEncoder
from ippai.simulate.tissue_properties import TissueProperties
from utils import AbsorptionSpectrum
import numpy as np


class IPPAISerializer(JSONEncoder):

    def default(self, _object: object):

        if isinstance(_object, TissueProperties.Chromophore):
            return _object.__dict__

        if isinstance(_object, AbsorptionSpectrum):
            return _object.__dict__

        if isinstance(_object, np.ndarray):
            return list(_object)

        if isinstance(_object, (np.int, np.int16, np.int32, np.int64)):
            return int(_object)

        if isinstance(_object, (np.float, np.float16, np.float32, np.float64)):
            return float(_object)

        return super().default(_object)
