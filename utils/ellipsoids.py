import numpy as np
import torch


def generate_random_3d_ellipsoids(
    device,
    diag_len: int,
    out_shape: tuple,
    n_ellipsoids: int = 1,
    axes_scale: float = 0.2,
    n_labels: int = 1,
    random_orientations: bool = True,
) -> torch.Tensor:
    """Generate a 3D tensor with random ellipsoids.

    Args:
        device (torch.device): The device to use.
        shape (tuple): The shape of the 3D tensor to generate.
        n_ellipsoids (int, optional): The number of ellipsoids to generate. Defaults to 1.
        n_labels (int, optional): The number of possible different labels for ellipsoids. Defaults to 1.
        random_orientations (bool, optional): Whether to apply random orientations for the ellipsoids. Defaults to True.

    Returns:
        torch.Tensor: The generated 3D tensor with random ellipsoids.
    """

    # Get the dimensions of the grid
    dim_x, dim_y, dim_z = (diag_len,) * 3

    with torch.device(device):
        # Initialise a tensor for the grid with zeros
        ellipsoids = torch.zeros(dim_x, dim_y, dim_z).long()

        # Create a 3D grid of coordinates
        x = torch.linspace(-1, 1, dim_x)
        y = torch.linspace(-1, 1, dim_y)
        z = torch.linspace(-1, 1, dim_z)
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")

        # Flatten the grid into a list for rotation application
        coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        # Precompute random values
        if random_orientations:

            # Generate random Euler angles for rotation for each ellipsoid
            theta_x = torch.rand(n_ellipsoids) * 2 * np.pi
            theta_y = torch.rand(n_ellipsoids) * 2 * np.pi
            theta_z = torch.rand(n_ellipsoids) * 2 * np.pi
        else:
            theta_x = theta_y = theta_z = np.zeros(n_ellipsoids)

        # Generate random radii
        a_vals, b_vals, c_vals = torch.normal(
            mean=axes_scale, std=0.2, size=(3, n_ellipsoids)
        )

        # Generate random centers for each ellipsoid
        centers = torch.rand((n_ellipsoids, 3)) - 0.5

        for idx in range(n_ellipsoids):
            # Rotation matrices for current ellipsoid
            Rx = torch.tensor(
                [
                    [1, 0, 0],
                    [0, torch.cos(theta_x[idx]), -torch.sin(theta_x[idx])],
                    [0, torch.sin(theta_x[idx]), torch.cos(theta_x[idx])],
                ]
            )

            Ry = torch.tensor(
                [
                    [torch.cos(theta_y[idx]), 0, torch.sin(theta_y[idx])],
                    [0, 1, 0],
                    [-torch.sin(theta_y[idx]), 0, torch.cos(theta_y[idx])],
                ]
            )

            Rz = torch.tensor(
                [
                    [torch.cos(theta_z[idx]), -torch.sin(theta_z[idx]), 0],
                    [torch.sin(theta_z[idx]), torch.cos(theta_z[idx]), 0],
                    [0, 0, 1],
                ]
            )

            # Combined rotation matrix
            R = Rz @ Ry @ Rx

            # Apply the rotation to the coordinates
            rotated_coords = coords @ R.T

            # Unflatten the rotated coordinates
            x_rot, y_rot, z_rot = (
                rotated_coords[:, 0].view(dim_x, dim_y, dim_z),
                rotated_coords[:, 1].view(dim_x, dim_y, dim_z),
                rotated_coords[:, 2].view(dim_x, dim_y, dim_z),
            )

            # Generate the ellipsoid using the equation of an 3D ellipsoid
            ellipsoid = (
                ((x_rot - centers[idx, 0]) ** 2) / a_vals[idx] ** 2
                + ((y_rot - centers[idx, 1]) ** 2) / b_vals[idx] ** 2
                + ((z_rot - centers[idx, 2]) ** 2) / c_vals[idx] ** 2
            ) <= 1

            # Add the ellipsoid to the space with the corresponding label
            ellipsoids[ellipsoid] = n_labels

            n_labels += 1

        # Crop the center region of the ellipsoids to match the desired shape
        start_idx = [(ellipsoids.shape[i] - out_shape[i]) // 2 for i in range(3)]
        end_idx = [start_idx[i] + out_shape[i] for i in range(3)]
        ellipsoids = ellipsoids[
            tuple(slice(start_idx[i], end_idx[i]) for i in range(3))
        ]

    return ellipsoids
