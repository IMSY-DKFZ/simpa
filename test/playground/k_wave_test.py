from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.models.acoustic_model import run_acoustic_forward_model
import matplotlib.pyplot as plt
import numpy as np
import os


random_seed = 12345
np.random.seed(random_seed)

settings = {
    Tags.SIMULATION_PATH: "/home/kris/hard_drive/data/k-wave/test_data",
    Tags.OPTICAL_MODEL_OUTPUT_NAME: "/home/kris/networkdrives/E130-Projekte/Photoacoustics/PreProcessedData/"
                                    "20190703_upsampling_experiment/multi_scale/training/Structure_0033014/"
                                    "spacing_0.15/optical_forward_model_output_885.npz",
    Tags.VOLUME_NAME: "test_data",
    Tags.RANDOM_SEED: random_seed,
    Tags.RUN_ACOUSTIC_MODEL: True,
    Tags.ACOUSTIC_MODEL_SCRIPT: "simulate",
    Tags.GPU: True,

    Tags.SPACING_MM: 0.15,
    Tags.MEDIUM_ALPHA_COEFF: 0.1,
    Tags.MEDIUM_ALPHA_POWER: 1.5,
    Tags.MEDIUM_SOUND_SPEED: 1500,

    Tags.SENSOR_RECORD: "p",
    Tags.SENSOR_CENTER_FREQUENCY: 7.5e6,
    Tags.SENSOR_BANDWIDTH: 133,
    Tags.SENSOR_DIRECTIVITY_ANGLE: 0,   # Most sensitive in x-dir (up/down)
    Tags.SENSOR_DIRECTIVITY_SIZE: 0.001,    # [m]
    Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",

    Tags.PMLInside: False,
    Tags.PMLSize: [20, 20],
    Tags.PMLAlpha: 2,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: True,
    Tags.MOVIENAME: "test"
}

os.makedirs(settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME], exist_ok=True)

acoustic_output_path = run_acoustic_forward_model(settings)

plt.subplot(121)
initital_pressure = np.load(settings[Tags.OPTICAL_MODEL_OUTPUT_NAME])["initial_pressure"]
plt.imshow(np.rot90(np.log10(initital_pressure), 3))
plt.subplot(122)
data = np.load(acoustic_output_path)["sensor_data"]
plt.imshow(np.rot90(np.flip(data, 0), 3))
plt.tight_layout()
plt.show()
