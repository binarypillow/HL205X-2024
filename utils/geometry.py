import array_api_compat.torch as xp
import matplotlib.pyplot as plt
import torch
from parallelproj import (
    RegularPolygonPETLORDescriptor,
    RegularPolygonPETProjector,
    RegularPolygonPETScannerGeometry,
    SinogramSpatialAxisOrder,
)


class PetGeometry:
    """PET scanner geometry class."""

    def __init__(
        self,
        device: torch.device,
        radius: float = 65,
        num_sides: int = 12,
        num_lor_endpoints_per_side: int = 8,
        lor_spacing: float = 4,
        distance_between_rings: float = 20,
        num_rings: int = 3,
        simmetry_axis: int = 2,
        radial_trim: int = 10,
        max_ring_difference: int = 2,
        img_shape: tuple = (147, 35, 147),
        voxel_size: tuple = (3.0, 2.0, 2.0),
    ):
        """Initialise the PET scanner geometry.

        Args:
            device (torch.device): The device to use.
            radius (float, optional): The radius of the scanner. Defaults to 65.
            num_sides (int, optional): The number of sides of the scanner. Defaults to 12.
            num_lor_endpoints_per_side (int, optional): The number of LOR endpoints per side. Defaults to 8.
            lor_spacing (float, optional): The spacing between LOR endpoints. Defaults to 4.
            distance_between_rings (float, optional): The distance between rings. Defaults to 20.
            num_rings (int, optional): The number of rings. Defaults to 3.
            simmetry_axis (int, optional): The symmetry axis. Defaults to 1.
            radial_trim (int, optional): The radial trim. Defaults to 10.
            max_ring_difference (int, optional): The maximum ring difference. Defaults to 2.
            img_shape (tuple, optional): The image shape. Defaults to (147, 35, 147).
            voxel_size (tuple, optional): The voxel size. Defaults to  (3.0, 2.0, 2.0).
        """
        # Create the PET scanner geometry
        self.scanner = RegularPolygonPETScannerGeometry(
            xp,
            str(device),
            radius=radius,
            num_sides=num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            ring_positions=torch.linspace(
                -distance_between_rings, distance_between_rings, num_rings
            ),
            symmetry_axis=simmetry_axis,
        )
        # Create the LOR descriptor
        self.lor_desc = RegularPolygonPETLORDescriptor(
            self.scanner,
            radial_trim=radial_trim,
            max_ring_difference=max_ring_difference,
            sinogram_order=SinogramSpatialAxisOrder.RVP,
        )
        # Create the projector
        self.proj = RegularPolygonPETProjector(
            self.lor_desc,
            img_shape=img_shape,
            voxel_size=voxel_size,
        )

    def show(self) -> None:
        """Show the PET scanner geometry.

        Displays:
            Two 3D plots showing the scanner geometry.
        """

        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        self.scanner.show_lor_endpoints(ax1)
        self.proj.show_geometry(ax2)
        plt.show()
