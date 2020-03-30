from simulate import Tags
from simulate.models.acoustic_models.acoustic_modelling import run_acoustic_forward_model
import matplotlib.pyplot as plt
import numpy as np
import os

optical_path = "/home/kris/hard_drive/data/pipeline_test/UpsamplingPhantom_200000/optical_forward_model_output_800.npz"

settings = {
    Tags.SIMULATION_PATH: "/home/kris/hard_drive/data/k-wave/test_data",
    Tags.VOLUME_NAME: "test_data",
    Tags.RUN_ACOUSTIC_MODEL: True,
    Tags.ACOUSTIC_MODEL_SCRIPT: "simulate",
    Tags.GPU: True,

    Tags.SPACING_MM: 0.15,
    Tags.MEDIUM_ALPHA_COEFF: 0.1,
    Tags.MEDIUM_ALPHA_POWER: 1.5,
    Tags.MEDIUM_SOUND_SPEED: "/home/kris/hard_drive/data/k-wave/test_data/test_data/sound_speed.npy",
    Tags.MEDIUM_DENSITY: "/home/kris/hard_drive/data/k-wave/test_data/test_data/medium_density.npy",

    Tags.SENSOR_MASK: "/home/kris/hard_drive/data/k-wave/test_data/test_data/sensor_mask.npy",
    Tags.SENSOR_RECORD: "p",
    Tags.SENSOR_CENTER_FREQUENCY_MHZ: 7.5e6,
    Tags.SENSOR_BANDWIDTH_PERCENT: 133,
    Tags.SENSOR_DIRECTIVITY_ANGLE: "/home/kris/hard_drive/data/k-wave/test_data/test_data/directivity_angle.npy", #0,   # Most sensitive in x-dir (up/down)
    Tags.SENSOR_DIRECTIVITY_SIZE_M: 0.001,    # [m]
    Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",

    Tags.PMLInside: False,
    Tags.PMLSize: [20, 20],
    Tags.PMLAlpha: 2,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: True,
    Tags.MOVIENAME: "test"
}

os.makedirs(settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME], exist_ok=True)

acoustic_output_path = run_acoustic_forward_model(settings, optical_path)

plt.subplot(121)
initital_pressure = np.load(optical_path)["initial_pressure"]
plt.imshow(np.rot90(np.log10(initital_pressure), 3))
plt.subplot(122)
data = np.load(acoustic_output_path)["sensor_data"]
plt.imshow(np.rot90(np.flip(data, 0), 3))
plt.tight_layout()
plt.show()
