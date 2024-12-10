import time
from functools import wraps

import torch
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def timeit(func):
    """Decorator to measure the time taken by a function to execute.

    Args:
        func (function): Function to measure the time taken to execute.

    Returns:
        function: Wrapper function that measures the time taken to execute the function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Execution time: {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def compute_metrics(
    device,
    true_img: torch.Tensor,
    predicted_img: torch.Tensor,
) -> dict:
    """Compute MSE, SSIM and PSNR between the provided images.

    Args:
        device (torch.device): Device to use.
        true_img (torch.Tensor): The ground truth image.
        predicted_img (torch.Tensor): The predicted image.

    Returns:
        dict: A dictionary containing lists of the computed metrics.
    """

    # Ensure the images have the same dimensions
    assert (
        true_img.shape == predicted_img.shape
    ), "Input images must have the same dimensions"

    # If the images are 2D, add a channel dimension
    if len(true_img.shape) == 2:
        true_img = torch.unsqueeze(true_img, 1)
        predicted_img = torch.unsqueeze(predicted_img, 1)

    mse = MeanSquaredError().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    metrics = {
        "mse": mse(
            predicted_img.unsqueeze(0).contiguous(), true_img.unsqueeze(0).contiguous()
        ),
        "ssim": ssim(predicted_img.unsqueeze(0), true_img.unsqueeze(0)),
        "psnr": psnr(predicted_img.unsqueeze(0), true_img.unsqueeze(0)),
    }

    return metrics
