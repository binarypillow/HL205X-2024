import os
import sys
import time

import torch

# Check if the argument is provided
if len(sys.argv) < 2:
    print("Error: Configuration file path not provided.")
    print(f"Usage: python {os.path.basename(__file__)} <config_file_path>")
    sys.exit(1)

# Variables
CHECK_INTERVAL = 30  # seconds
FREE_VRAM_THRESHOLD = 0.8  # Fraction of VRAM that must be free


def get_free_vram_fraction():
    """Returns the fraction of free VRAM on the first GPU."""

    if not torch.cuda.is_available():
        print("No GPU detected. Exiting.")
        sys.exit(1)

    free_memory = torch.cuda.mem_get_info()[0]
    total_memory = torch.cuda.mem_get_info()[1]
    free_fraction = free_memory / total_memory

    return free_fraction


def main():
    print("Monitoring VRAM usage...")

    while True:
        free_vram = get_free_vram_fraction()
        print(f"Free VRAM fraction: {free_vram:.2f}")

        if free_vram >= FREE_VRAM_THRESHOLD:
            print("Sufficient free VRAM detected. Running the script...")
            os.system(f"python train_models.py {sys.argv[1]}")
            break  # Exit after running the script

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
