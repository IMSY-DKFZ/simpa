# k-Wave-Python migration validation

## Purpose

This change validates and completes the migration of SIMPA acoustic forward
simulation and time-reversal reconstruction from MATLAB k-Wave to
k-wave-python 0.4.0.

The implementation is based on Faberno's original migration commit:

d7fba581 - switch from k-Wave (matlab) to k-Wave-python

## Production changes

### k_wave_adapter.py

- Avoid passing scalar sound speed, density, and attenuation values to
  `np.pad`.
- Convert 2D detector orientations from radians to degrees at the
  `kWaveArray.add_rect_element` API boundary.
- Add one grid-spacing coordinate correction in every modeled (2D/3D) direction.
- Retain the three-voxel gel padding only on padded 2D solver-axis 0.
- Remove gel-layer compensation from unpadded 3D geometry.
- Translate supported SIMPA `ADDITIONAL_FLAGS` to k-wave-python execution
  options.
- Forward the configured k-Wave binary path.
- Select the CUDA or OMP executable according to the SIMPA GPU setting.

### time_reversal_adapter.py

- Remove MATLAB's legacy `+1` sensor-mask indexing shift.
- Reject negative and out-of-grid detector coordinates.
- Apply the inverse k-Wave angular permutation to SIMPA detector rows.
- Validate that detector ordering forms a complete permutation.
- Forward binary paths and additional execution flags.
- Preserve k-wave-python's reconstruction axis order.
- Correct exception handling for `ValueError` and `KeyError`.

### Supporting changes

- Disable logger propagation to prevent duplicate output.
- Make path-manager tests snapshot and restore real configuration files.
- Prevent automatic tests from leaving `path_config.env` renamed as
  `path_config.env.backup`.

## Automatic tests added

### KWave geometry

- 2D gel and material-map padding.
- Scalar homogeneous medium properties.
- Radian-to-degree detector orientation.
- 3D coordinate correction in every direction.
- Nonempty 2D boundary-detector mask.
- Nonempty 3D boundary-detector mask.

### Time reversal

- Complete solver handoff.
- Inverse angular detector permutation.
- Invalid permutation rejection.
- Zero-based sensor-mask coordinates.
- Negative/out-of-grid coordinate rejection.
- Native reconstruction axis order.

### Execution configuration

- Additional-flag translation and forwarding.
- Binary-path forwarding.
- OMP selection with `GPU=False`.
- CUDA selection with `GPU=True`.

## Numerical validation

| Area | Result |
| --- | --- |
| Optical-to-acoustic handoff | PASS: correlation approximately 1.0; relative L2 approximately 1e-7 |
| 2D forward, equivalent inputs | PASS: relative L2 approximately 2.4e-6 |
| 2D time reversal, identical inputs/masks | PASS: correlation 0.9999999771; relative L2 4.36e-4; zero peak shift |
| 3D forward, equivalent geometry | PASS: relative L2 approximately 2.6e-7 |
| 3D detector ordering | PASS: all 256 traces retain identity and zero sample lag |
| Full-pipeline initial pressure | PASS: relative L2 8.0e-8 |
| Full-pipeline time series | PASS: relative L2 1.04e-6; identity ordering; zero sample lag |
| Full-pipeline reconstruction | PASS after documented legacy `[+1,+1]` MATLAB alignment: correlation 0.9961 |

The approximately 5% difference previously observed in the production 2D
forward result comes from correcting a legacy SIMPA unit error: radians were
passed to an API that expects degrees. It is not a MATLAB/Python solver
incompatibility.

## Test results

Focused migration tests:

    Ran 19 tests
    OK

Complete automatic suite:

    Ran 177 tests
    OK (expected failures=27)

## Validation commands

Create and activate a Python 3.10 environment:

    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install "torch==2.11.0"
    python -m pip install -e ".[testing]"

Run focused migration tests:

    python -m unittest \
      simpa_tests.automatic_tests.test_additional_flags \
      simpa_tests.automatic_tests.test_k_wave_geometry \
      simpa_tests.automatic_tests.test_time_reversal_adapter \
      -v

Run the full automatic suite:

    python -m unittest discover \
      -s simpa_tests\automatic_tests \
      -p "test*.py" \
      -v

Run the manual overview:

    python simpa_tests\manual_tests\generate_overview.py

## Local path configuration

A local `path_config.env` is required:

    SIMPA_SAVE_DIRECTORY=C:/path/to/output
    MCX_BINARY_PATH=C:/path/to/mcx.exe
    MATLAB_BINARY_PATH=C:/path/to/.venv/Lib/site-packages/kwave/bin/windows/kspaceFirstOrder-omp.exe

`MATLAB_BINARY_PATH` retains its historical name for compatibility, but points
to the k-wave-python native executable. The adapter switches between OMP and
CUDA according to the GPU setting.

## Known manual-test observations

These remain open but are not evidence of k-Wave solver incompatibility:

- PointSourceReconstruction row three differs visually.
- QPAIReconstruction stops after eight iterations while the reference shows ten.
- TestLinearUnmixingVisual has a diamond-shaped absolute-difference region.
- MATLABAdditionalFlags expects a MATLAB-generated logfile and needs a
  backend-agnostic replacement.
- Two reconstruction tests can encounter Windows HDF5 locking.
- The infinitesimal-slab test has malformed JSON reference data.

The migration should therefore be described as validated for the tested 2D and
3D SIMPA configurations, rather than as proof of every possible geometry,
medium, or hardware configuration.