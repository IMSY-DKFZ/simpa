from json import JSONEncoder
from utils import AbsorptionSpectrum, Chromophore
import numpy as np


class IPPAISerializer(object):
    """
    TODO
    """

    def serialize(self, _object: object):
        """

        """
        if isinstance(_object, Chromophore):
            return _object.__dict__

        if isinstance(_object, AbsorptionSpectrum):
            return _object.__dict__

        return None


class IPPAIJSONSerializer(JSONEncoder):
    """
    TODO
    """

    def __init__(self):
        """
        TODO
        """
        super().__init__()
        self._serializer = IPPAISerializer()

    def default(self, _object: object):
        """
        TODO
        """

        serialized_object = self._serializer.serialize(_object)

        if serialized_object is not None:
            return serialized_object

        if isinstance(_object, np.ndarray):
            return list(_object)

        if isinstance(_object, (np.int, np.int16, np.int32, np.int64)):
            return int(_object)

        if isinstance(_object, (np.float, np.float16, np.float32, np.float64)):
            return float(_object)

        return super().default(_object)
