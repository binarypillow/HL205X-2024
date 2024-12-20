import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sinogram(
    sinogram: np.ndarray,
    num_planes: int = 20,
) -> None:
    """Display a subset of planes from a sinogram.

    Args:
        sinogram (np.ndarray): The sinogram to display.
        num_planes (int, optional): The number of planes to display. Defaults to 20.

    Displays:
        A figure with a grid of subplots, each containing a plane from the sinogram.
    """

    # Calculate the number of rows and columns
    cols = 7
    rows = math.ceil(num_planes / cols)

    _, ax = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    ax = ax.ravel()

    for i in range(rows * cols):
        if i < num_planes:
            ax[i].imshow(
                sinogram[:, :, i].T,
                cmap="grey",
            )
            ax[i].set_title(f"sino plane {i}", fontsize="medium")
        else:
            ax[i].axis("off")

    # Hide the axes
    plt.tight_layout()
    plt.show()


def plot_vol_slices(
    slices,
    num_slices=20,
) -> None:
    """Display a subset of planes from a sinogram.

    Args:
        sinogram (torch.Tensor): The slices to display.
        num_planes (int, optional): The number of slices to display. Defaults to 20.

    Displays:
        A figure with a grid of subplots, each containing a plane from the sinogram.
    """

    # Calculate the number of rows and columns
    cols = 7
    rows = math.ceil(num_slices / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    ax = ax.ravel()

    for i in range(rows * cols):
        if i < num_slices:
            ax[i].imshow(
                slices[:, i, :],
                cmap="grey",
                vmin=0,
                vmax=1.5,
            )
            ax[i].set_title(f"Slice {i}", fontsize="medium")
        else:
            ax[i].axis("off")

    # Hide the axes
    plt.tight_layout()
    plt.show()


def plot_images(image_pairs: list, n_col=4, fig_dim=None) -> None:
    """Display a grid of images with their corresponding titles.

    Args:
        image_pairs (list): A list of tuples, each containing:
            - img (np.ndarray): The image to be displayed.
            - title (str): The title of the subplot.
            - data_range (tuple): The range of pixel values in the image.
        n_col (int): Number of columns in the grid. Default is 4.
        fig_dim (tuple): Figure dimensions. If None, the default dimensions are used.

    Displays:
        A figure with a grid of subplots, each containing the images from the list.
    """

    # Calculate the number of rows required
    n_images = len(image_pairs)
    n_row = math.ceil(n_images / n_col)

    # Set figure dimensions
    if fig_dim is None:
        fig_dim = (2 * n_col, 2 * n_row)

    fig, ax = plt.subplots(n_row, n_col, figsize=fig_dim)
    ax = np.array(ax).flatten()  # Flatten to 1D for easier indexing

    # Plot each image and set title
    for i, (img, title, data_range) in enumerate(image_pairs):
        if data_range is None:
            data_range = (img.min(), img.max())
        im = ax[i].imshow(img, cmap="gray", vmin=data_range[0], vmax=data_range[1])
        ax[i].set_title(title, fontsize="medium")
        ax[i].axis("off")
        # Create a colorbar for the current image
        fig.colorbar(
            im,
            ax=ax[i],
            fraction=0.05,
            pad=0.05,
            shrink=0.75,
            ticks=np.arange(data_range[0], data_range[1] + 1, 0.3),
        )

    # Hide any remaining unused subplots
    for j in range(len(image_pairs), len(ax)):
        ax[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_comparison(
    true_img: np.ndarray,
    predicted_img: np.ndarray,
    title1: str = "True Image",
    title2: str = "Predicted Image",
) -> None:
    """Compare two images by displaying them side by side along with their difference.

    Args:
        predicted_img (np.ndarray): The reconstructed image.
        true_img (np.ndarray): The true image.
        title1 (str, optional): The title for the true image. Defaults to "True Image".
        title2 (str, optional): The title for the predicted image. Defaults to "Predicted Image".

    Displays:
        A figure with three subplots: the predicted image, the true image, and the difference between them.
    """

    # Ensure the images have the same dimensions
    assert (
        true_img.shape == predicted_img.shape
    ), "Input images must have the same dimensions"

    fig = plt.figure(figsize=(15, 5))

    # Predicted Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(true_img, cmap="grey", vmin=0, vmax=1)
    ax1.set_title(title1)
    ax1.axis("off")

    # True Image
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(predicted_img, cmap="grey", vmin=0, vmax=1)
    ax2.set_title(title2)
    ax2.axis("off")

    # Difference Image
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(true_img - predicted_img, cmap="grey", vmin=-0.5, vmax=0.5)
    ax3.set_title("Difference")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def plot_slice(img: np.ndarray, axis: bool = True, dual: bool = False) -> None:
    """Display a single slice of an image or sinogram.
    Args:
        img (np.ndarray): The image or sinogram to display.
        dual (bool): Whether to plot the dual domain. Default is False.

    Displays:
        A figure with a single subplot containing the image.
    """

    plt.figure(figsize=(12, 6))
    plt.imshow(img, cmap="grey")
    if dual:
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$t$")
    else:
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
    if axis:
        plt.axis("on")
    else:
        plt.axis("off")
    plt.show()


def plot_3d_views(img: np.ndarray) -> None:
    """Plot 3D views of a volume.

    Args:
        img (np.ndarray): 3D volume to be plotted.

    Displays:
        A figure with three 3D planes.
    """
    # Calculate the central slices for each view
    central_transverse = img[img.shape[0] // 2, :, :]
    central_sagittal = img[:, img.shape[1] // 2, :]
    central_coronal = img[:, :, img.shape[2] // 2]

    # Plot the central slices
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(central_transverse.T, cmap="grey", vmin=0, vmax=1.5)
    ax1.set_title("Transverse Plane")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(central_sagittal, cmap="grey", vmin=0, vmax=1.5)
    ax2.set_title("Sagittal View")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(central_coronal, cmap="grey", vmin=0, vmax=1.5)
    ax3.set_title("Coronal Plane")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def plot_loss_metrics(
    loss_epoch: list,
    psnr_epoch: list,
    ssim_epoch: list,
) -> None:
    """Plot the loss and metric values across epochs.

    Args:
        loss_epoch (list): A list of loss values across epochs.
        psnr_epoch (list): A list of PSNR values across epochs.
        ssim_epoch (list): A list of SSIM values across epochs.

    Displays:
        A figure with three subplots: the training loss, PSNR, and SSIM across epochs.
    """

    fig = plt.figure(figsize=(15, 5))

    # Loss
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(loss_epoch)
    ax1.grid(True)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # PSNR
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(psnr_epoch)
    ax2.grid(True)
    ax2.set_title("Training PSNR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")

    # SSIM
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(ssim_epoch)
    ax3.grid(True)
    ax3.set_title("Training SSIM")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Metric")

    plt.tight_layout()
    plt.show()


def plot_metrics(list_metrics) -> None:
    """Plot the loss and metric values for different noise levels with standard deviation.

    Args:
        list_metrics (list of tuples): Each tuple contains:
            - A tuple of 3 lists of tuples (MSE, SSIM, PSNR each as (mean, std) across epochs).
            - A label for the corresponding model.

    Displays:
        A figure with three subplots: the MSE, PSNR, and SSIM for different noise levels.
    """
    x = np.arange(0.5, 10.0, 0.5).tolist()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for metrics, label in list_metrics:
        mse_epoch, ssim_epoch, psnr_epoch = metrics

        # Unpack mean and standard deviation for each metric
        mse_mean, mse_std = mse_epoch
        ssim_mean, ssim_std = ssim_epoch
        psnr_mean, psnr_std = psnr_epoch

        # MSE
        axs[0].plot(x, mse_mean, linestyle="--", marker="o", label=label)
        axs[0].fill_between(
            x,
            np.array(mse_mean) - np.array(mse_std),
            np.array(mse_mean) + np.array(mse_std),
            alpha=0.2,
        )
        axs[0].set_ylim(0, 0.03)
        axs[0].grid(True)
        axs[0].set_title("MSE")
        axs[0].set_xlabel("Noise Level")
        axs[0].set_ylabel("Value")

        # SSIM
        axs[1].plot(x, ssim_mean, linestyle="--", marker="o", label=label)
        axs[1].fill_between(
            x,
            np.array(ssim_mean) - np.array(ssim_std),
            np.array(ssim_mean) + np.array(ssim_std),
            alpha=0.2,
        )
        axs[1].set_ylim(0.3, 1)
        axs[1].grid(True)
        axs[1].set_title("SSIM")
        axs[1].set_xlabel("Noise Level")
        axs[1].set_ylabel("Value")

        # PSNR
        axs[2].plot(x, psnr_mean, linestyle="--", marker="o", label=label)
        axs[2].fill_between(
            x,
            np.array(psnr_mean) - np.array(psnr_std),
            np.array(psnr_mean) + np.array(psnr_std),
            alpha=0.2,
        )
        axs[2].set_ylim(15, 25)
        axs[2].grid(True)
        axs[2].set_title("PSNR")
        axs[2].set_xlabel("Noise Level")
        axs[2].set_ylabel("Value")

    # Add legends to each subplot
    for ax in axs:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_iterations_output(
    device: torch.device,
    model: torch.nn.Module,
    x: torch.Tensor,
    idx_slice: int,
    cab: bool = False,
    dual: bool = False,
) -> None:
    """Capture and plot the outputs at each iteration of the LPD model.

    Args:
        device (torch.device): The device to use.
        model (torch.nn.Module): The model to use.
        x (torch.Tensor): The input tensor (sinogram).
        idx_slice (int): The index of the slice to plot.
        cab (bool): Whether to plot the outputs of the CAB layers. Default is False.
        dual (bool): Whether to plot the outputs of the dual layers. Default is False.

    Displays:
        A figure with subplots of the intermediate outputs at each iteration.
    """
    # Dictionary to store intermediate outputs
    intermediate_outputs = []

    # Hook function to capture intermediate outputs
    def hook_fn(module, input, output):
        """Hook function to capture the output of a layer."""
        intermediate_outputs.append((module, output))

    # Register hooks
    if cab:
        layers = model.dual_cabs if dual else model.primal_cabs
    else:
        layers = model.dual_layers if dual else model.primal_layers

    for layer in layers:
        layer.register_forward_hook(hook_fn)

    model.eval()
    # Perform the forward pass
    with torch.no_grad():
        x = x.to(device)

        with torch.autocast(device_type=str(device)):
            out = model(x.unsqueeze(1))

    # Plot the intermediate outputs
    num_layers = (
        len(intermediate_outputs) if dual or cab else len(intermediate_outputs) + 1
    )
    _, ax = plt.subplots(1, num_layers, figsize=(15, 5))
    ax = ax.ravel()

    i = 0
    intermediate_outs = (
        reversed(intermediate_outputs) if not cab else intermediate_outputs
    )
    for i, (_, output) in enumerate(intermediate_outs):
        ax[i].imshow(output.squeeze(1)[idx_slice, :, :].cpu().numpy(), cmap="gray")
        ax[i].set_title(f"Layer {i + 1}")
        ax[i].axis("off")

    if not dual and not cab:
        ax[i + 1].imshow(out.squeeze(0)[:, idx_slice, :].cpu().numpy(), cmap="gray")
        ax[i + 1].set_title("Output")
        ax[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_iterations_output_3d(
    device: torch.device,
    model: torch.nn.Module,
    x: torch.Tensor,
    idx_slice: int,
    dual: bool = False,
) -> None:
    """Capture and plot the outputs at each iteration of the LPD model. Modified for 3D volumes.

    Args:
        device (torch.device): The device to use.
        model (torch.nn.Module): The model to use.
        x (torch.Tensor): The input tensor (sinogram).
        idx_slice (int): The index of the slice to plot.
        dual (bool): Whether to plot the outputs of the dual layers. Default is False.

    Displays:
        A figure with subplots of the intermediate outputs at each iteration.
    """
    # Dictionary to store intermediate outputs
    intermediate_outputs = []

    # Hook function to capture intermediate outputs
    def hook_fn(module, input, output):
        """Hook function to capture the output of a layer."""
        intermediate_outputs.append((module, output))

    # Register hooks
    layers = model.dual_layers if dual else model.primal_layers

    for layer in layers:
        layer.register_forward_hook(hook_fn)

    model.eval()
    # Perform the forward pass
    with torch.no_grad():
        x = x.to(device)

        with torch.autocast(device_type=str(device)):
            _ = model(x.unsqueeze(1))

    # Plot the intermediate outputs
    num_layers = len(intermediate_outputs)
    _, ax = plt.subplots(1, num_layers, figsize=(15, 5))
    ax = ax.ravel()

    i = 0
    for i, (_, output) in enumerate(intermediate_outputs):
        (
            ax[i].imshow(
                output.squeeze().squeeze()[:, idx_slice, :].cpu().numpy(), cmap="gray"
            )
            if not dual
            else ax[i].imshow(
                output.squeeze().squeeze()[:, :, idx_slice].cpu().numpy(), cmap="gray"
            )
        )
        ax[i].set_title(f"Layer {i + 1}")
        ax[i].axis("off")

    plt.tight_layout()
    plt.show()
