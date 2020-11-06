# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
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

import numpy as np

def assert_equals_recursive(a, b):
    if isinstance(a, dict):
        for item in a:
            assert item in a, (str(item) + " was not in a: " + str(a))
            assert item in b, (str(item) + " was not in b: " + str(b))
            if isinstance(a[item], dict):
                assert_equals_recursive(a[item], b[item])
            elif isinstance(a[item], list):
                assert_equals_recursive(a[item], b[item])
            else:
                if isinstance(a[item], np.ndarray):
                    assert (a[item] == b[item]).all()
                else:
                    assert a[item] == b[item], str(a[item]) + " is not the same as " + str(b[item])
    elif isinstance(a, list):
        for item1, item2 in zip(a, b):
            assert_equals_recursive(item1, item2)
    else:
        assert a == b, str(a) + " is not the same as " + str(b)