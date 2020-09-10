# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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

import unittest
from simpa.core import TissueProperties
from simpa.utils import TISSUE_LIBRARY
from simpa.utils import SPECTRAL_LIBRARY
from simpa.utils import calculate_oxygenation
from simpa.utils.serialization import SIMPAJSONSerializer
import json
import os
import numpy as np


class TestTissueProperties(unittest.TestCase):

    def setUp(self):
        print("\n[SetUp]")

    def tearDown(self):
        print("\n[TearDown]")

    def test_find_absorption_spectra(self):
        settings = TISSUE_LIBRARY.blood_arterial()
        assert isinstance(settings, dict)
        tp = TissueProperties(settings=settings)
        assert tp is not None and isinstance(tp, TissueProperties)
        [mua, mus, g] = tp.get(798)
        assert mua is not None and isinstance(mua, float)
        assert mus is not None and isinstance(mus, float)
        assert g is not None and isinstance(g, float)

    def test_absorption_spectra_interpolation(self):
        for absorption_spectrum in SPECTRAL_LIBRARY:
            for i in range(600, 900, 1):
                wl_lower = absorption_spectrum.get_absorption_for_wavelength(i)
                wl_upper = absorption_spectrum.get_absorption_for_wavelength(i+1)
                wl_in_between = absorption_spectrum.get_absorption_for_wavelength(i+0.5)
                if wl_upper > wl_lower:
                    assert (wl_in_between < wl_upper) and (wl_in_between > wl_lower)
                elif wl_lower > wl_upper:
                    assert (wl_in_between > wl_upper) and (wl_in_between < wl_lower)
                elif np.abs(wl_lower - wl_upper) < 1e-5:
                    assert ((np.abs(wl_lower - wl_in_between) < 1e-5) and (np.abs(wl_lower - wl_in_between) < 1e-5))
                else:
                    raise ValueError("Ranges undefined!")

    def test_oxygenation_calculation(self):
        for i in range(100):
            tp = TissueProperties(settings=TISSUE_LIBRARY.blood_generic())
            oxy = calculate_oxygenation(tp)
            assert ((oxy >= 0) and (oxy <= 1))

            tp = TissueProperties(settings=TISSUE_LIBRARY.blood_arterial())
            oxy = calculate_oxygenation(tp)
            assert ((oxy >= 0) and (oxy <= 1))

            tp = TissueProperties(settings=TISSUE_LIBRARY.blood_venous())
            oxy = calculate_oxygenation(tp)
            assert ((oxy >= 0) and (oxy <= 1))

    def test_get_constant_settings(self):
        for i in range(100):
            mua = np.random.random() * 10
            mus = np.random.random() * 100
            g = np.random.random()

            settings = TISSUE_LIBRARY.constant(mua=mua, mus=mus, g=g)
            tp = TissueProperties(settings=settings)
            for wl in range(700, 900, 5):
                [_mua, _mus, _g] = tp.get(wl)
                assert np.abs(mua - _mua) < 1e-5
                assert np.abs(mus - _mus) < 1e-5
                assert np.abs(g - _g) < 1e-5

    def test_dictionary_serialization(self):
        settings = TISSUE_LIBRARY.blood_arterial()
        tmp_json_filename = "test_settings.json"

        serializer = SIMPAJSONSerializer()

        with open(tmp_json_filename, "w") as json_file:
            json.dump(settings, json_file, indent="\t", default=serializer.default)

        with open(tmp_json_filename, "r+") as json_file:
            new_settings = json.load(json_file)

        print(new_settings)

        os.remove(tmp_json_filename)

