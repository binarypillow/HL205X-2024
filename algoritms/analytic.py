from typing import Literal

import torch
from parallelproj import RegularPolygonPETProjector

from utils.metrics import timeit


class Filter:
    """Filter for the filtered back projection algorithm. Supports Ramp and Hamming filters."""

    def __init__(self, filter_type: Literal["Ramp", "Hamming"] = "Ramp") -> None:
        """Initialise the filter.

        Args:
            filter_type (str, optional): The type of filter to use. Defaults to "Ramp".
                                         Options are "Ramp" or "Hamming".
        """
        # Check if the filter type is supported
        assert filter_type in ["Ramp", "Hamming"], "Unsupported filter type."
        self.filter_type = filter_type

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        """Apply the selected filter.

        Args:
            w (torch.Tensor): The frequency domain.

        Returns:
            torch.Tensor: The filtered frequency domain.
        """
        if self.filter_type == "Ramp":

            # Return the absolute value of the frequency domain
            return torch.abs(w)

        elif self.filter_type == "Hamming":

            # Compute the Hamming window
            alpha = 0.54
            beta = 1 - alpha
            N = len(w)
            hamming_window = alpha - beta * torch.cos(2 * torch.pi * w / (N - 1))
            # Return the Hamming window multiplied by the absolute value of the frequency domain
            return hamming_window * torch.abs(w)
        else:
            raise ValueError("Unsupported filter type.")


class FBP:
    """Filtered back projection (FBP) algorithm."""

    def __init__(
        self,
        projector: RegularPolygonPETProjector,
        device: torch.device,
        filter: Filter,
    ) -> None:
        """Initialise the FBP algorithm.

        Args:
            projector (RegularPolygonPETProjector): The projector.
            device (torch.device): The device to use.
            filter (Filter, optional): The filter to use. Defaults to RampFilter.
        """
        self.proj = projector
        self.device = device
        self.filter = filter

    @timeit
    def __call__(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Run the filtered back projection algorithm.

        Args:
            sinogram (torch.Tensor): The sinogram.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        # Create the frequency domain filter
        freq_fft = (
            torch.fft.fftfreq(sinogram.shape[-2]).reshape((-1, 1)).to(self.device)
        )

        filter = self.filter(freq_fft)
        # Compute the Fourier transform of the sinogram
        sino_fft = torch.fft.fft2(sinogram, dim=(0, 1))

        # Apply the filter
        filtered_sino_fft = filter * sino_fft

        # Compute the inverse Fourier transform
        filtered_sino = torch.fft.ifft2(filtered_sino_fft, dim=(0, 1)).real
        # Compute the back projection
        prediction = self.proj.adjoint(filtered_sino)

        print("FBP algorithm finished.")
        return prediction
