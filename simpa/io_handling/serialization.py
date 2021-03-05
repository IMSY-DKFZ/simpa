# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from json import JSONEncoder
from simpa.utils import AbsorptionSpectrum, Molecule
import numpy as np


class SIMPASerializer(object):
    """
    TODO
    """

    def serialize(self, _object: object):
        """

        """
        if isinstance(_object, Molecule):
            return _object.__dict__

        if isinstance(_object, AbsorptionSpectrum):
            return _object.__dict__

        return _object


class SIMPAJSONSerializer(JSONEncoder):
    """
    TODO
    """

    def __init__(self):
        """
        TODO
        """
        super().__init__()
        self._serializer = SIMPASerializer()

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
