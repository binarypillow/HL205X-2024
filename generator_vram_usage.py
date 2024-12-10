import time

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from configs._scanners import get_miniPET_geometry
from generators.mixed import MixedDataset
from utils.inits import init_loguru, init_pytorch

# Initialise the device
device = init_pytorch()

# Initialise the logger and get the output path
output_path, file_name = init_loguru(
    save_to_file=True,
    config_name="_dataMeasures",
)

# Global variables
SAVE_OUTPUT: bool = False
IMG_WIDTH: int = 147
IMG_HEIGHT: int = 147
IMG_DEPTH: int = 35
NOISE_LEVEL: float = 0.5
BATCH_SIZE: int = 6
TRAIN_SAMPLES: int = 500
NUM_LABELS: int = 6
NUM_BATCHES: int = 25

# Create the miniPET geometry
miniPET_geometry = get_miniPET_geometry(
    device=device, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, num_rings=IMG_DEPTH
)
synthetic_dataset = MixedDataset(
    device,
    projector=miniPET_geometry.proj,
    n_imgs=TRAIN_SAMPLES,
    n_labels=6,
    noise_interval=(0.1, 1.2),
    kernel_size=5,
    sigma=2.0,
)
data_loader = DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Lists to store timing and VRAM measurements
timing_results = []
vram_before_results = []
vram_after_results = []
vram_peak_results = []

data_loader_iter = iter(data_loader)

for _ in range(NUM_BATCHES):
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated(device)
    else:
        vram_before = 0

    # Load batch
    sinogram_shapes, image_shapes, label_shapes, map_shapes = next(data_loader_iter)

    end_time = time.perf_counter()
    duration = end_time - start_time
    timing_results.append(duration)

    del sinogram_shapes, image_shapes, label_shapes, map_shapes

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_after = torch.cuda.memory_allocated(device)
        vram_peak = torch.cuda.max_memory_allocated(device)
    else:
        vram_after = vram_peak = 0

    # Convert VRAM usage to GB and store results
    vram_before_results.append(vram_before / (1024**3))
    vram_after_results.append(vram_after / (1024**3))
    vram_peak_results.append(vram_peak / (1024**3))

# Calculate mean and standard deviation
timing_mean = np.mean(timing_results)
timing_std = np.std(timing_results)
vram_before_mean = np.mean(vram_before_results)
vram_before_std = np.std(vram_before_results)
vram_after_mean = np.mean(vram_after_results)
vram_after_std = np.std(vram_after_results)
vram_peak_mean = np.mean(vram_peak_results)
vram_peak_std = np.std(vram_peak_results)

# Output results
logger.info(f"Timing Mean (s): {timing_mean}")
logger.info(f"Timing Std (s): {timing_std}")
logger.info(f"VRAM Before Mean (GB): {vram_before_mean}")
logger.info(f"VRAM Before Std (GB): {vram_before_std}")
logger.info(f"VRAM After Mean (GB): {vram_after_mean}")
logger.info(f"VRAM After Std (GB): {vram_after_std}")
logger.info(f"VRAM Peak Mean (GB): {vram_peak_mean}")
logger.info(f"VRAM Peak Std (GB): {vram_peak_std}")
