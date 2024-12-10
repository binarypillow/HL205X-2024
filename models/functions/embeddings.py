import math

import torch
import torch.nn.functional as F
from parallelproj import RegularPolygonPETProjector


def sinogram_patching(
    device: torch.device, projector: RegularPolygonPETProjector, patch_size: int = 7
) -> torch.Tensor:
    """Generate the indices and values of non-zero elements in the sinogram of each patch.

    Args:
        torch.device: The device to use.
        projector (RegularPolygonPETProjector): The projector.
        patch_size (int, optional): The size of the patch. Defaults to 3.

    Returns:
        torch.Tensor: The stacked tensor of indices and values
    """
    # Get the shape of the input image (height, width, depth)
    image_shape = projector.in_shape

    # Calculate the number of patches along one dimension and total number of patches
    patch_dim = int(math.ceil(image_shape[0] / patch_size))  # Patches per dimension
    patch_num = patch_dim**2  # Total number of patches

    # Create a tensor to hold the patches
    images = torch.zeros(patch_num, image_shape[0], 1, image_shape[2]).to(device)

    # Initialise lists to store indices and values of non-zero elements in the sinogram
    indices = []
    values = []
    sinos = []

    # Loop over each patch
    for k in range(patch_num):
        # Determine the i, j coordinates of the current patch in the patch grid
        i = k // patch_dim  # Row index
        j = k % patch_dim  # Column index

        # Fill the current patch with ones
        images[
            k,
            i * patch_size : (i + 1) * patch_size,
            0,
            j * patch_size : (j + 1) * patch_size,
        ] = 1

        # Project the current patch and store the sinogram
        sino = projector(images[k])
        sinos.append(sino)

        # Get the non-zero indices from the sinogram and store them
        index = (sino > 0).squeeze(-1).nonzero()
        indices.append(index)

    # Calculate the maximum length of indices to ensure proper padding later
    lengths = [len(tensor) for tensor in indices]
    max_length = max(lengths)

    # Pad tensors to the maximum length and compute values if needed
    padded_indices = []
    for i, idx in enumerate(indices):
        pad_size = max_length - len(idx)
        padded_idx = F.pad(
            idx, (0, 0, pad_size, 0)
        )  # Pad tensor along the first dimension
        padded_indices.append(padded_idx)

        # Use the respective sinogram to compute the values
        values.append(
            sinos[i][padded_idx[:, 0], padded_idx[:, 1], 0]
            / torch.sum(sinos[i][padded_idx[:, 0], padded_idx[:, 1], 0])
        )

    # Stack the padded index and the values tensors
    padded_indices = torch.stack(padded_indices)
    values = torch.stack(values)

    return torch.cat((padded_indices, values.unsqueeze(-1)), dim=2)


def multi_sinogram_patching(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    patch_size: int = 7,
    embed_dim: int = 256,
) -> torch.Tensor:
    """Generate the indices and values of non-zero elements in the sinogram of each patch.
        The function creates overlapping patches of a fixed size in each sinogram.

    Args:
        torch.device: The device to use.
        projector (RegularPolygonPETProjector): The projector.
        patch_size (int, optional): The size of the patch. Defaults to 7.
        embed_dim (int, optional): The embedding dimension. Defaults to 256.

    Returns:
        torch.Tensor: The stacked tensor of indices and values
    """
    # Get the shape of the input image (height, width, depth)
    image_shape = projector.in_shape

    # Calculate the number of patches along one dimension and total number of patches
    patch_dim = int(math.ceil(image_shape[0] / patch_size))  # Patches per dimension
    patch_num = patch_dim**2  # Total number of patches

    # Create a tensor to hold the patches
    images = torch.zeros(patch_num, image_shape[0], 1, image_shape[2]).to(device)

    # Initialise lists to store indices and values of non-zero elements in the sinogram
    indices = []
    sinos = []
    values = []
    patches = []

    for k in range(patch_num):
        i = k // patch_dim  # Row index
        j = k % patch_dim  # Column index

        # Fill the current patch with ones
        images[
            k,
            i * patch_size : (i + 1) * patch_size,
            0,
            j * patch_size : (j + 1) * patch_size,
        ] = 1

        # Project the current patch and store the sinogram
        sino = projector(images[k])

        # Get the non-zero indices from the sinogram and store them
        index = (sino != 0).squeeze(-1).nonzero()
        sorted_index = index[index[:, 1].argsort()]
        indices.append(sorted_index)

        sinos.append(sino)

    # Create patches of equal size in each sinogram
    for i, idx in enumerate(indices):
        total_pixels = len(idx)
        if total_pixels <= embed_dim:
            raise ValueError(
                "The number of non-zero pixels in the sinogram is less than the target number of pixels per patch."
            )
        else:
            # Create overlapping patches
            stride = (total_pixels - embed_dim) // (
                (total_pixels + embed_dim - 1) // embed_dim - 1
            )
            start_indices = range(0, total_pixels - embed_dim + 1, max(1, stride))
            for start in start_indices:
                patch = idx[start : start + embed_dim]
                patches.append(patch)

                values.append(
                    sinos[i][patch[:, 0], patch[:, 1], 0]
                    / torch.sum(sinos[i][patch[:, 0], patch[:, 1], 0])
                )

    # Stack the patches indices and the values tensors
    patches = torch.stack(patches)
    values = torch.stack(values)

    return torch.cat((patches, values.unsqueeze(-1)), dim=2)
