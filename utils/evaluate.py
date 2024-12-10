import torch

from utils.metrics import timeit


@timeit
def evaluate_model(
    device: torch.device,
    model: torch.nn.Module,
    sinos: torch.Tensor,
):
    """Evaluate the model on the provided sinograms.

    Args:
        device (torch.device): The device to use.
        model (torch.nn.Module): The model to evaluate.
        sinos (torch.Tensor): The sinograms to evaluate the model on.

    Returns:
        torch.Tensor: The reconstructed images.
    """

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        sinos = sinos.to(device)

        # Forward pass
        with torch.autocast(device_type=str(device)):
            recon_imgs = model(sinos.unsqueeze(1))

    return recon_imgs.squeeze(0)
