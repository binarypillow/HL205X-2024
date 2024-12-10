import os
import sys
import time

import torch
import yaml
from loguru import logger
from torch import nn


def init_pytorch() -> torch.device:
    """Identify the available device(s) and returns it(them).

    Returns:
        torch.device: A tuple containing the device and the number of GPUs.
    """
    try:
        # Check which device is available and set it accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda")
            num_gpus = torch.cuda.device_count()
            print(f"Using CUDA with {num_gpus} GPU(s)...")
            # Empty the cache to avoid memory issues
            torch.cuda.empty_cache()

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders)...")
        else:
            device = torch.device("cpu")
            print("Using CPU...")
    # If an error occurs, default to CPU
    except Exception as e:
        device = torch.device("cpu")
        print(f"Error initializing device: {e}. Defaulting to CPU.")

    # Set the seed for reproducibility
    ## 42 is the Answer to the Ultimate Question of Life, the Universe and Everything
    torch.manual_seed(42)

    return device


def get_vram() -> str:
    """Get the available VRAM.

    Returns:
        str: The used VRAM / the total VRAM
    """
    string = "| VRAM: N/A"
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1024**3
        total = torch.cuda.mem_get_info()[1] / 1024**3
        string = f"| VRAM: {total - free:.2f}/{total:.2f}GB"
    return string


def init_weights(m: torch.nn.Module) -> None:
    """Initialise the weights of the model using a normal distribution with mean 0 and standard deviation 0.01.

    Args:
        m (torch.nn.Module): The model to initialise the weights for.

    Returns:
        None: The weights of the model are initialised.
    """

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Initialise weights to a normal distribution with mean 0 and standard deviation 0.01
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

        # Initialise bias to a small constant value (0.01)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def init_loguru(
    save_to_file: bool = False,
    config_name: str = "config.yml",
) -> tuple[str, str]:
    """Initialise the logger and save the logs to a file if required.

    Args:
        save_to_file (bool, optional): Whether to save the logs to a file. Defaults to False.
        config_name (str, optional): The name of the configuration file. Defaults to "config.yml".

    Returns:
        str: The path to the log file.
    """

    # Remove any existing logger
    logger.remove()

    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

    # Add the logger to the console and the file
    logger.add(
        sink=sys.stderr,
        level="DEBUG",
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    # Log the initialisation
    logger.info("Initialised logger.")

    output_path = f"outputs/{config_name}"

    file_name = time.strftime("%Y%m%d_%H%M")

    if save_to_file:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        log_file = f"{output_path}/{file_name}_logs.log"

        logger.add(
            sink=log_file,
            level="DEBUG",
            format=log_format,
            colorize=False,
            backtrace=True,
            diagnose=True,
        )
        # Log the log file path
        logger.info(f"Saving logs to {log_file}...")

    return output_path, file_name


def load_config(config_file: str) -> dict:
    """Load the configuration file and returns the parameters.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: The parameters loaded from the configuration file.
    """
    # Check if the file exists
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    return config


class EarlyStopper:
    """Class to implement early stopping based on the loss."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialise the EarlyStopper.

        Args:
            patience (int, optional): The number of epochs to wait before stopping. Defaults to 10.
            min_delta (float, optional): The minimum change in loss to consider. Defaults to 0.0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf")

    def early_stop(self, loss: float, model: torch.nn.Module, path: str) -> bool:
        """Check if the training should be stopped based on the loss.

        Args:
            loss (float): The loss value to check.
            model (torch.nn.Module): The model to save if the training should be stopped.
            path (str): The path to save the model.

        Returns:
            bool: Whether the training should be stopped.
        """
        # If the loss is less than the minimum loss, save the model
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
            # Save the model
            torch.save(model.state_dict(), path)

        # If difference between the loss and the minimum loss is less than the minimum delta, increment the counter
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop the training
        return False
