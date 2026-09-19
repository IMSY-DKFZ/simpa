# SPDX-FileCopyrightText: 2026 Erol Tasci
# SPDX-License-Identifier: MIT

import numpy as np

from simpa import Spectrum


def test_spectrum_tensors_can_be_passed_to_another_spectrum_constructor():
    wavelengths = np.array([400, 500, 600])
    values = np.array([0.1, 0.2, 0.3])
    original = Spectrum("original", wavelengths, values)

    copied = Spectrum("copied", original.wavelengths, original.values)

    np.testing.assert_array_equal(copied.wavelengths.numpy(), wavelengths)
    np.testing.assert_array_equal(copied.values.numpy(), values)
