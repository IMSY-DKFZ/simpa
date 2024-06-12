# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os

profile_type = os.getenv("SIMPA_PROFILE")
if profile_type is None:
    # define a no-op @profile decorator
    def profile(f):
        return f
elif profile_type == "TIME":
    import atexit
    from line_profiler import LineProfiler

    profile = LineProfiler()
    atexit.register(profile.print_stats)
elif profile_type == "MEMORY":
    from memory_profiler import profile
elif profile_type == "GPU_MEMORY":
    from pytorch_memlab import profile
    from torch.cuda import memory_summary
    import atexit

    @atexit.register
    def print_memory_summary():
        print(memory_summary())
else:
    raise RuntimeError("SIMPA_PROFILE env var is defined but invalid: valid values are TIME, MEMORY, or GPU_MEMORY")
