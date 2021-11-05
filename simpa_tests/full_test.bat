coverage run --source="simpa" -m pytest automatic_tests/
coverage run -a --source="simpa" manual_tests/processing_components/TestLinearUnmixingVisual.py
coverage run -a --source="simpa" manual_tests/processing_components/QPAIReconstruction.py
coverage run -a --source="simpa" manual_tests/test_with_experimental_measurements/ReproduceDISMeasurements.py
coverage run -a --source="simpa" manual_tests/optical_forward_models/AbsorptionAndScatteringWithinHomogenousMedium.py
coverage run -a --source="simpa" manual_tests/optical_forward_models/AbsorptionAndScatteringWithInifinitesimalSlabExperiment.py
coverage run -a --source="simpa" manual_tests/optical_forward_models/CompareMCXResultsWithDiffusionTheory.py
coverage run -a --source="simpa" manual_tests/acoustic_forward_models/MinimalKWaveTest.py



coverage report
coverage html -d ../docs/full_coverage/