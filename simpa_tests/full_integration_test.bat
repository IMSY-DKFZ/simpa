coverage run --source="simpa" -m pytest automatic_tests/
coverage run -a --source="simpa" manual_tests/processing_components/TestLinearUnmixingVisual.py
coverage run -a --source="simpa" manual_tests/processing_components/QPAIReconstruction.py
coverage run -a --source="simpa" manual_tests/test_with_experimental_measurements/ReproduceDISMeasurements.py
coverage run -a --source="simpa" manual_tests/optical_forward_models/AbsorptionAndScatteringWithinHomogenousMedium.py
coverage run -a --source="simpa" manual_tests/optical_forward_models/AbsorptionAndScatteringWithInifinitesimalSlabExperiment.py
coverage run -a --source="simpa" manual_tests/optical_forward_models/CompareMCXResultsWithDiffusionTheory.py
coverage run -a --source="simpa" manual_tests/acoustic_forward_models/MinimalKWaveTest.py
coverage run -a --source="simpa" manual_tests/acoustic_forward_models/KWaveAcousticForwardConvenienceFunction.py
coverage run -a --source="simpa" manual_tests/image_reconstruction/DelayAndSumReconstruction.py
coverage run -a --source="simpa" manual_tests/image_reconstruction/DelayMultiplyAndSumReconstruction.py
coverage run -a --source="simpa" manual_tests/image_reconstruction/SignedDelayMultiplyAndSumReconstruction.py
coverage run -a --source="simpa" manual_tests/image_reconstruction/TimeReversalReconstruction.py
coverage run -a --source="simpa" manual_tests/digital_device_twins/VisualiseDevices.py
coverage run -a --source="simpa" manual_tests/digital_device_twins/SimulationWithMSOTInvision.py
coverage run -a --source="simpa" manual_tests/volume_creation/SegmentationLoader.py


coverage report
coverage html -d ../docs/full_coverage/