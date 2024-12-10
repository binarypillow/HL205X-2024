from copy import copy

import torch
from parallelproj import LinearOperatorSequence, RegularPolygonPETProjector

from utils.metrics import timeit


class MLEM:
    """Maximum Likelihood Expectation Maximization (MLEM) algorithm."""

    def __init__(
        self,
        projector: RegularPolygonPETProjector,
        device: torch.device,
        n_iterations: int = 10,
    ) -> None:
        """Initialise the MLEM algorithm.

        Args:
            projector (RegularPolygonPETProjector): The projector.
            device (torch.device): The device to use.
            n_iterations (int, optional): The number of iterations. Defaults to 10.
        """
        self.proj = projector
        self.device = device
        self.n_iterations = n_iterations

    @timeit
    def __call__(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Run the MLEM algorithm.

        Args:
            sinogram (torch.Tensor): The sinogram.

        Returns:
            torch.Tensor: The reconstructed image.
        """

        # Get the image and sinogram shapes
        img_shape = self.proj.in_shape
        sino_shape = self.proj.out_shape

        # Create the initial prediction as a tensor of ones
        prediction = torch.ones(img_shape).to(self.device)

        # Compute the backward projection of ones
        adjoint_ones = self.proj.adjoint(torch.ones(sino_shape).to(self.device))

        for _ in range(self.n_iterations):
            # Compute the forward projection of the prediction
            foward_proj = self.proj(prediction)
            # Compute the ratio between the sinogram and the previous forward projection
            ratio = sinogram / (foward_proj + 1e-9)
            # Compute the backward projection of the ratio
            backward_proj = self.proj.adjoint(ratio)
            # Normalise and update the prediction
            prediction *= backward_proj / adjoint_ones

        print("MLEM algorithm finished.")
        return prediction


class OSEM:
    """Ordered Subset Expectation Maximization (OSEM) algorithm."""

    def __init__(
        self,
        projector: RegularPolygonPETProjector,
        device: torch.device,
        n_iterations: int = 10,
        n_subsets: int = 4,
    ) -> None:
        """Initialise the OSEM algorithm.

        Args:
            projector (RegularPolygonPETProjector): The projector.
            device (torch.device): The device to use.
            n_iterations (int, optional): The number of iterations. Defaults to 10.
            n_subsets (int, optional): The number of subsets. Defaults to 4.
        """
        self.proj = projector
        self.device = device
        self.n_iterations = n_iterations
        self.n_subsets = n_subsets

    @timeit
    def __call__(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Run the OSEM algorithm.

        Args:
            sinogram (torch.Tensor): The sinogram.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        # Get the image and sinogram shapes
        img_shape = self.proj.in_shape
        sino_shape = self.proj.out_shape

        # Get the distributed views and slices
        subset_views, subset_slices = (
            self.proj.lor_descriptor.get_distributed_views_and_slices(
                self.n_subsets, len(sino_shape)
            )
        )

        # Clear the cached LOR endpoints since we will create many copies of the projector
        self.proj.clear_cached_lor_endpoints()
        subset_proj_seq = []

        # Create the projector for each subset
        for i in range(self.n_subsets):
            subset_proj = copy(self.proj)
            subset_proj.views = subset_views[i]

            subset_proj_seq.append(subset_proj)

        # Create a sequence of linear operators, one for each subset
        subset_proj_seq = LinearOperatorSequence(subset_proj_seq)

        # Create the initial prediction as a tensor of ones
        prediction = torch.ones(img_shape).to(self.device)

        # Compute a list of backward projection of ones, one for each subset
        subset_adjoint_ones = [
            subset_proj.adjoint(torch.ones(subset_proj.out_shape).to(self.device))
            for subset_proj in subset_proj_seq
        ]

        for _ in range(self.n_iterations):
            for k, sl in enumerate(subset_slices):
                # Compute the forward projection of the subset prediction
                foward_proj = subset_proj_seq[k](prediction)
                # Compute the ratio between the subset sinogram and the previous forward projection
                ratio = sinogram[sl] / (foward_proj + 1e-9)
                # Compute the backward projection of the ratio
                backward_proj = subset_proj_seq[k].adjoint(ratio)
                # Normalise and update the prediction
                prediction *= backward_proj / subset_adjoint_ones[k]

        print("OSEM algorithm finished.")
        return prediction
