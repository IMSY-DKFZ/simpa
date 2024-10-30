# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from enum import Enum


# region Public enums

class MCXVolumeBoundaryCondition(Enum):
    """Defines the behavior of the photons touching the volume boundaries.

    Sets the letters of the --bc command (https://mcx.space/wiki/index.cgi?Doc/mcx_help).
    Note: The behavior is only defined on the volume faces in the x- and y-axis, the behavior for the faces
    on the z-axis always remains the default.
    """

    DEFAULT = "______000000"
    """The default behavior."""
    MIRROR_REFLECTION = "mm_mm_000000"
    """The photons are totally reflected as if the volume faces are mirrors."""
    CYCLIC = "cc_cc_000000"
    """The photons reenter from the opposite volume face."""
    ABSORB = "aa_aa_000000"
    """The photons are fully absorbed."""
    FRESNEL_REFLECTION = "rr_rr_000000"
    """The photons are reflected based on the Fresnel equations."""

# endregion
