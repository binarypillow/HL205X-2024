"""
   Adapted from https://github.com/matt-kh/synthmorph-torch
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F


def draw_perlin(
    device: torch.device,
    out_shape: tuple,
    scales: list | int,
    min_std: float = 0.1,
    max_std: float = 1.0,
    dtype: torch.dtype = torch.float32,
    seed=None,
) -> torch.Tensor:
    """Generate Perlin noise by drawing from Gaussian distributions at different resolutions.

    Parameters:
        device (torch.device): Device to use.
        out_shape (tuple): Shape of the output tensor.
        scales (list | int): List of scales for the noise.
        min_std (float, optional): Minimum standard deviation for the noise.
            Defaults to 0.1.
        max_std (float, optional): Maximum standard deviation for the noise.
            Defaults to 1.0.
        dtype (torch.dtype, optional): Data type for the noise. Defaults to torch.float32.
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        torch.Tensor: The generated Perlin noise.
    """
    out_shape_np = np.asarray(out_shape, dtype=np.int32)
    if isinstance(scales, (int)):
        scales = [scales]

    rand = np.random.default_rng(seed)

    def _seed():
        return int(rand.integers(np.iinfo(int).max))

    rng = torch.Generator(device=device).manual_seed(_seed())

    out = torch.zeros(out_shape, dtype=dtype, device=device)
    for scale in scales:
        sample_shape = np.ceil(out_shape_np[:-1] / scale)
        sample_shape = np.asarray((*sample_shape, out_shape_np[-1]), dtype=np.int32)

        std = torch.empty(size=(), dtype=dtype, device=device).uniform_(
            min_std, max_std, generator=rng
        )
        gauss = torch.empty(
            size=tuple(sample_shape), dtype=dtype, device=device
        ).normal_(std=std.item(), generator=rng)

        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        out += gauss if scale == 1 else resize(device, gauss, zoom[:-1])

    # Transform to Torch format
    indices = list(range(len(out.shape)))
    out = out.permute(-1, *indices[:-1])

    return out


@torch.no_grad()
def labels_to_image(
    device: torch.device,
    labels: torch.Tensor,
    out_shape=None,
    num_chan: int = 1,
    one_hot: bool = True,
    out_label_list=None,
) -> dict:
    """Augment label maps and synthesize images from them.

    Parameters:
        device (torch.device): Device to use.
        labels (torch.Tensor): Label maps to synthesise images from.
        out_shape (optional): List of the spatial dimensions of the outputs.
            Inputs will be symmetrically cropped or zero-padded to fit. Defaults to the input shape.
        num_chan (optional): Number of image channels to be synthesized. Defaults to 1.
        one_hot (bool, optional): Whether output label maps are one-hot encoded.
            Only the specified output labels will be included. Defaults to False.
        out_label_list (optional): List of labels to include in the output label maps.
    """

    # Compute the number of labels in the input label maps.
    n_labels = len(torch.unique(labels))

    n_dim = len(labels.shape)

    in_shape = labels.shape
    if out_shape is None:
        out_shape = in_shape
    in_shape, out_shape = map(np.asarray, (in_shape, out_shape))

    # Add new axes, Torch format
    labels = labels.unsqueeze(0).unsqueeze(1)
    labels = labels.expand(1, -1, *[-1] * n_dim)

    # Transform labels into [0, 1, ..., N-1].
    labels = labels.to(dtype=torch.int32, device=device)
    in_label_list = labels.unique()
    num_in_labels = len(in_label_list)

    in_lut = torch.zeros(
        size=(int(torch.max(in_label_list).item()) + 1,),
        dtype=torch.int32,
        device=device,
    )
    for i, lab in enumerate(in_label_list):
        in_lut[lab] = i
    labels = in_lut[labels]
    labels = torch_to_tf(labels)

    labels = labels.to(torch.int32)

    intensities = torch.linspace(0, 255, n_labels).to(device)

    # Intensity manipulations: add noise to the intensities
    diff = torch.diff(intensities)[0] // 3
    intensities += (torch.rand(n_labels).to(device) * 2 * diff) - diff

    # Synthetic image generation
    image = torch.zeros(size=labels.shape, device=device)
    indices = torch.concat(
        [labels + i * num_in_labels for i in range(num_chan)], dim=-1
    )

    def gather(x):
        return torch.reshape(x[0], (-1,))[x[1]]

    # Apply the mapping to the labels
    values = gather([intensities, indices])
    image = image + values

    # Intensity manipulations
    image = torch.clamp(image, min=0, max=255)
    image = torch.stack([minmax_norm(batch) for batch in image])
    image = tf_to_torch(image).squeeze(0)

    # Lookup table for converting the index labels back to the original values,
    # setting unwanted labels to background. If the output labels are provided
    # as a dictionary, it can be used e.g. to convert labels to GM, WM, CSF
    if out_label_list is None:
        out_label_list = in_label_list
    if isinstance(out_label_list, (tuple, list, torch.Tensor)):
        out_label_list = {lab.item(): lab for lab in out_label_list}

    out_lut = torch.zeros((num_in_labels,), dtype=torch.int32)
    for i, lab in enumerate(in_label_list):
        if lab.item() in out_label_list:
            out_lut[i] = out_label_list[lab.item()]

    # For one-hot encoding, update the lookup table such that the M desired
    # output labels are rebased into the interval [0, M-1[. If the background
    # with value 0 is not part of the output labels, set it to -1 to remove it
    # from the one-hot maps
    if one_hot:
        hot_label_list = torch.tensor(list(out_label_list.values())).unique()  # Sorted
        hot_lut = torch.full(
            (hot_label_list[-1] + 1,), fill_value=-1, dtype=torch.int32, device=device
        )
        for i, lab in enumerate(hot_label_list):
            hot_lut[lab] = i
        out_lut = hot_lut[out_lut]

    # Convert indices to output labels only once.
    labels = out_lut[labels]
    if one_hot:
        labels = F.one_hot(labels.to(torch.int64), num_classes=len(hot_label_list))
        labels = tf_to_torch(labels.squeeze(-2))
    else:
        labels = labels.squeeze(-1)

    all_outputs = [image, labels]
    image, labels = [i.squeeze(0) for i in all_outputs]

    outputs = {"image": image, "labels": labels, "mapping": torch.unique(image)}

    return outputs


def resize(
    device: torch.device,
    vol: torch.Tensor,
    zoom_factor: list | int,
    interp_method: str = "linear",
) -> torch.Tensor:
    """Resise a volume by a given zoom factor.

    Args:
        device (torch.device): The device to use.
        vol (torch.Tensor): The input volume to resize.
        zoom_factor (list | int): The zoom factor for each dimension.
        interp_method (str, optional): The interpolation method. Defaults to "linear".

    Returns:
        torch.Tensor: The resized volume.
    """
    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (
            ndims,
            ndims + 1,
        ), "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)

    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims

    # Skip resize for zoom_factor of 1
    if all(z == 1 for z in zoom_factor):
        return vol

    if not isinstance(vol_shape[0], int):
        vol_shape = list(vol_shape)

    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [torch.linspace(0.0, vol_shape[d] - 1.0, new_shape[d]) for d in range(ndims)]
    grid = ndgrid(*lin)
    grid = [g.to(device) for g in grid]

    return interpn(device, vol, grid, interp_method=interp_method)


def interpn(
    device: torch.device,
    vol: torch.Tensor,
    loc: list | torch.Tensor,
    interp_method: str = "linear",
) -> torch.Tensor:
    """Interpolate a volume at given locations.

    Args:
        device (torch.device): The device to use.
        vol (torch.Tensor): The input volume to interpolate.
        loc (list | torch.Tensor): The locations at which to interpolate the volume.
        interp_method (str, optional): The interpolation method. Defaults to "linear".

    Returns:
        torch.Tensor: The interpolated volume.
    """
    vol = vol.to(device)
    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, dim=-1).to(device)
    nb_dims = loc.shape[-1]
    input_vol_shape = vol.shape

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception(
            "Number of loc Tensors %d does not match volume dimension %d"
            % (nb_dims, len(vol.shape[:-1]))
        )

    if nb_dims > len(vol.shape):
        raise Exception(
            "Loc dimension %d does not match volume dimension %d"
            % (nb_dims, len(vol.shape))
        )

    if len(vol.shape) == nb_dims:
        vol = torch.unsqueeze(vol, -1)

    # Flatten and float location tensors
    if not loc.dtype.is_floating_point:
        target_loc_dtype = vol.dtype if vol.dtype.is_floating_point else torch.float32
        loc = loc.to(target_loc_dtype)
    elif vol.dtype.is_floating_point and vol.dtype != loc.dtype:
        loc = loc.to(vol.dtype)

    if isinstance(vol.shape, torch.Size):
        vol_shape = list(vol.shape)
    else:
        vol_shape = vol.shape

    max_loc = [d - 1 for d in list(vol.shape)]

    if interp_method == "linear":
        loc0 = loc.floor()

        # Clip values
        clipped_loc = [loc[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [loc0[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]

        # Get other end of point cube
        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.to(torch.int32) for f in loc0lst], [f.to(torch.int32) for f in loc1]]

        # Compute the difference between the upper value and the original value
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]

        # Note reverse ordering since weights are inverse of diff.
        weights_loc = [diff_loc1, diff_loc0]

        # Go through all the cube corners, indexed by a ND binary vector
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))

        interp_vol = torch.zeros(
            list(loc[..., 0].shape) + [vol_shape[-1]], device=device
        )

        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]
            idx = sub2ind2d(vol_shape[:-1], subs)
            vol_reshape = torch.reshape(vol, [-1, vol_shape[-1]])
            vol_val = vol_reshape[idx.to(torch.int64)]

            # Get the weight of this cube_pt based on the distance to the loc
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)
            wt = wt.unsqueeze(-1)
            # Compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == "nearest", (
            "method should be linear or nearest, got: %s" % interp_method
        )
        roundloc = loc.round().to(torch.int32)
        roundloc = [roundloc[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]

        idx = sub2ind2d(vol_shape[:-1], roundloc)
        interp_vol = vol.reshape(-1, vol_shape[-1])[idx]

    if len(input_vol_shape) == nb_dims:
        assert interp_vol.shape[-1] == 1, "Something went wrong with interpn channels"
        interp_vol = interp_vol[..., 0]

    return interp_vol


def sub2ind2d(siz: list, subs: list, **kwargs) -> torch.Tensor:
    """Convert subscripts to linear indices in 2D.

    Args:
        siz (list): The size of the volume.
        subs (list): The subscripts to convert.

    Returns:
        torch.Tensor: The linear indices
    """
    # Subs is a list
    assert len(siz) == len(subs), "found inconsistent siz and subs: %d %d" % (
        len(siz),
        len(subs),
    )

    k = np.cumprod(siz[::-1])
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


def prod_n(lst: list) -> torch.Tensor:
    """Compute the product of a list of Tensors.

    Args:
        lst (list): The list of Tensors to multiply.

    Returns:
        torch.Tensor: The product of the Tensors.
    """
    prod = lst[0].clone()
    for p in lst[1:]:
        prod *= p
    return prod


def ndgrid(*args, **kwargs) -> tuple:
    """
    Broadcast Tensors on an N-D grid with ij indexing

    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)

    Returns:
        A list of Tensors
    """
    return torch.meshgrid(*args, indexing="ij", **kwargs)


def torch_to_tf(inp: torch.Tensor) -> torch.Tensor:
    """Convert a tensor from PyTorch format to Tensorflow format.

    Args:
        inp (torch.Tensor): The input tensor

    Returns:
        torch.Tensor: The converted tensor
    """

    indices = np.arange(inp.ndim)
    return inp.permute(0, *indices[2:], 1)


def tf_to_torch(inp: torch.Tensor) -> torch.Tensor:
    """Convert a tensor from TensorFlow format to PyTorch format.

    Args:
        inp (torch.Tensor): The input tensor

    Returns:
        torch.Tensor: The converted tensor
    """
    indices = np.arange(inp.ndim)
    return inp.permute(0, -1, *indices[1:-1])


def minmax_norm(x: torch.Tensor, axis=None) -> torch.Tensor:
    """
    Min-max normalize tensor using a safe division.
    Arguments:
        x (torch.Tensor): Input tensor.
        axis (int): Dimensions to reduce during normalization. If None, all axes will be
            considered, treating the input as a single image. To normalize batches or features
            independently, exclude the respective dimensions.
    Returns:
        torch.Tensor: Normalized tensor.
    """

    if axis is None:
        # Treated as fattened, 1D tensor
        def torchmin(x):
            return torch.min(x)

        def torchmax(x):
            return torch.max(x)

    else:
        # Operates on specified axis, and maintain shape
        def torchmin(x):
            return torch.min(x, dim=axis, keepdim=True).values

        def torchmax(x):
            return torch.max(x, dim=axis, keepdim=True).values

    x_min = torchmin(x)
    x_max = torchmax(x)
    result = torch.where(
        (x_max - x_min) != 0, (x - x_min) / (x_max - x_min), torch.zeros_like(x)
    )
    return result
