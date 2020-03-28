from utils import SPECTRAL_LIBRARY

def calculate_oxygenation(tissue_properties):
    """
    :return: an oxygenation value between 0 and 1 if possible, or None, if not computable.
    """

    hb = None
    hbO2 = None

    for chromophore in tissue_properties.chromophores:
        if chromophore.spectrum == SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN:
            hb = chromophore.volume_fraction
        if chromophore.spectrum == SPECTRAL_LIBRARY.OXYHEMOGLOBIN:
            hbO2 = chromophore.volume_fraction

    if hb is None and hbO2 is None:
        return None

    if hb is None:
        hb = 0
    elif hbO2 is None:
        hbO2 = 0

    if hb + hbO2 < 1e-10:  # negative values are not allowed and division by (approx) zero
        return None        # will lead to negative side effects.

    return hbO2 / (hb + hbO2)
