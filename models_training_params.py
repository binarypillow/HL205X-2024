import glob

import numpy as np
from loguru import logger

from utils.inits import init_loguru

path_list: list = [
    "outputs/lpdUnet2D_3/*_logs.log",
    "outputs/lpdUnetTransformer2D_1/*_logs.log",
    "outputs/lpdUnet3D_1/*_logs.log",
    "outputs/crossSinogramLpdUnet2D_1/*_logs.log",
    "outputs/crossImageLpdUnet2D_1/*_logs.log",
    "outputs/crossUpdateLpdUnet2D_1/*_logs.log",
    "outputs/crossConcatLpdUnet2D_1/*_logs.log",
]

# Initialise the logger and get the output path
output_path, file_name = init_loguru(
    save_to_file=True,
    config_name="_trainingParams",
)

for path in path_list:
    epochs_list = []
    vram_used_list = []

    for file_name in glob.glob(path):
        # Extract timestamps for INFO logs
        info_timestamps = []
        with open(file_name, "r") as file:
            logs = file.readlines()
            last_row = -1
            if "Early stopping activated!" in logs[last_row]:
                last_row = -2
            last_epoch = float(logs[last_row].split(" | ")[2].split("/")[0])
            vram_used = float(
                logs[last_row - 1].split(" | ")[-1].split(" ")[1].split("/")[0]
            )

        epochs_list.append(last_epoch)
        vram_used_list.append(vram_used)

    # Compute mean and standard deviation
    epochs_mean = np.mean(epochs_list)
    epochs_std = np.std(epochs_list)
    vram_mean = np.mean(vram_used_list)
    vram_std = np.std(vram_used_list)

    name = path.split("/")[1]

    # Display the results
    logger.info(
        f"{name} -> {vram_mean:.2f} +/- {vram_std:.2f} GB {epochs_mean:.2f} +/- {epochs_std:.2f} epochs"
    )
