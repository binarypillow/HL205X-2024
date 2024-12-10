import os

import nibabel as nib
import numpy as np
import odl
import torch
from einops import rearrange
from parallelproj import RegularPolygonPETProjector
from torchvision.transforms import functional as F


def load_miniPET_sino(device: torch.device, FILE_PATH: str) -> torch.Tensor:
    """Load a miniPET sinogram from a NIfTI file.

    Args:
        device (torch.device): The device to use.
        FILE_PATH (str): The path to the NIfTI file.

    Returns:
        torch.Tensor: The miniPET sinogram.
    """
    # Load the sinogram from file
    sino = torch.Tensor(nib.load(FILE_PATH).get_fdata()).to(device)

    # Resize and rearrange the sinogram
    sino_resized = F.resize(sino, [111, 210])
    sino_out = rearrange(sino_resized, "a b c -> b c a")

    return sino_out


def load_miniPET_img(device: torch.device, FILE_PATH: str) -> torch.Tensor:
    """Load a miniPET image from a NIfTI file and adjust it to match the reconstructed image.

    Args:
        device (torch.device): The device to use.
        FILE_PATH (str): The path to the NIfTI file.

    Returns:
        torch.Tensor: The miniPET image.
    """

    # Load the image from file
    img = torch.Tensor(nib.load(FILE_PATH).get_fdata() / 1000).to(device)
    img = F.resize(img, [147, 147])

    ### Adjust the miniPET image to match the reconstructed image
    # Rotate the image
    img = F.rotate(img, angle=-15)
    # Zoom in by cropping the center of the image
    center_crop_size = 135
    img = F.center_crop(img, output_size=[center_crop_size, center_crop_size])
    # Pad with zeros to the original size
    margin = (147 - center_crop_size) // 2
    img = F.pad(img, padding=[margin, margin, margin, margin])

    # Flip the image upside down
    img = torch.flip(img, [1, 2])

    # Apply the rearrange operation
    img = rearrange(img, "a b c -> b a c")

    return img


def save_tensor_to_nii(img: torch.Tensor, FILE_PATH: str) -> None:
    """Save a pytorch tensor to a NIfTI file.

    Args:
        img (torch.Tensor): The image to save.
        FILE_PATH (str): The path to save the image to.

    Returns:
        None
    """

    # If the directory does not exist, create it
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

    # Remove negative values
    img = torch.clamp(img, 0)

    # Apply the rearrange operation and convert to numpy
    img = rearrange(img, "a b c -> b a c").float().cpu().numpy()

    # Create an affine matrix as identity matrix (no transformation)
    affine_matrix = np.eye(4)
    img_ni = nib.Nifti1Image(img, affine=affine_matrix)
    nib.save(img_ni, FILE_PATH)


def create_SLPhantom(
    device: torch.device,
    IMG_WIDTH: int,
    IMG_HEIGHT: int,
    IMG_DEPTH: int,
    projector: RegularPolygonPETProjector,
    noise_level: float = 0.5,
    seed: bool = True,
) -> tuple:
    """Create the 3D image and sinogram of the Sheep Logan Phantom.

    Args:
        device (torch.device): The device to use.
        IMG_WIDTH (int): The width of the image.
        IMG_HEIGHT (int): The height of the image.
        IMG_DEPTH (int): The depth of the image.
        projector (RegularPolygonPETProjector): The projector.
        noise_level (float, optional): The noise level. Defaults to 0.5.
        seed (bool, optional): Whether to set the seed for reproducibility. Defaults to True.

    Returns:
        Tuple: The sinogram and the images.
    """

    # Create a 3D uniform space
    uniform_space: odl.DiscretizedSpace = odl.uniform_discr(
        min_pt=(0.0, -20.0, -20.0),
        max_pt=(0.5, 20.0, 20.0),
        shape=(
            IMG_WIDTH,
            IMG_HEIGHT,
            IMG_DEPTH * 2,
        ),  # double the depth to have enough space for the central slices
        dtype="float64",
    )

    # Create a 3D Shepp-Logan phantom and convert it to a PyTorch tensor
    sl_phantom: odl.DiscretizedSpaceElement = odl.phantom.shepp_logan(
        space=uniform_space, modified=True
    )

    # Get the central slices of the Shepp-Logan phantom
    slice_start = IMG_DEPTH - IMG_DEPTH // 2 - 1
    slice_end = IMG_DEPTH + IMG_DEPTH // 2
    img: torch.Tensor = rearrange(
        torch.tensor(sl_phantom, dtype=torch.float64).to(device).flip(1),
        "w h d -> h d w",
    )[:, slice_start:slice_end, :]

    # Normalise the Shepp-Logan phantom
    img = torch.clamp(img, 0, 1)

    if seed:
        # Set the seed for reproducibility
        torch.manual_seed(42)

    # Add Gaussian blur to the image
    img_blur = F.gaussian_blur(img, kernel_size=[5, 5], sigma=[2, 2])

    # Foward project the Shepp-Logan phantom to obtain the sinogram
    sino: torch.Tensor = projector(img_blur)
    sino = torch.poisson(sino / noise_level) * noise_level

    return sino, img
