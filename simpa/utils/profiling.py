# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os

profile_type = os.getenv("SIMPA_PROFILE")
if os.getenv("SIMPA_PROFILE_SAVE_FILE"):
    stream = open(os.getenv("SIMPA_PROFILE_SAVE_FILE"), 'w')
else:
    stream = None
if profile_type is None:
    # define a no-op @profile decorator
    def profile(f):
        return f
elif profile_type == "TIME":
    import atexit
    from line_profiler import LineProfiler

    profile = LineProfiler()

    def print_stats_atexit():
        return profile.print_stats(stream=stream, output_unit=10**(-3))
    atexit.register(print_stats_atexit)
elif profile_type == "MEMORY":
    from memory_profiler import profile
    profile = profile(stream=stream)
elif profile_type == "GPU_MEMORY":
    from typing import Tuple
    from pytorch_memlab.line_profiler.line_profiler import LineProfiler, DEFAULT_COLUMNS
    import atexit

    global_line_profiler = LineProfiler()
    global_line_profiler.enable()

    def profile(func, columns: Tuple[str, ...] = DEFAULT_COLUMNS):
        """
        Check plagiarism problems
        """
        import atexit
        global_line_profiler.add_function(func)

        def print_stats_atexit():
            global_line_profiler.print_stats(func, columns, stream=stream)

        atexit.register(print_stats_atexit)
        return func

else:
    raise RuntimeError("SIMPA_PROFILE env var is defined but invalid: valid values are TIME, MEMORY, or GPU_MEMORY")
