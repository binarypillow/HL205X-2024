import torch
import torch.utils.checkpoint as cp
from einops.layers.torch import Rearrange
from parallelproj import RegularPolygonPETProjector
from torch import nn

from models.functions.projector import TorchProjector


class CrossUpdateLPD(nn.Module):
    """The Learned Primal-Dual (LPD) algorithm that uses
    cross-attention blocks instead of updates."""

    def __init__(
        self,
        n_iter: int,
        projector: RegularPolygonPETProjector,
        primal_layers: nn.ModuleList,
        dual_layers: nn.ModuleList,
        primal_cabs: nn.ModuleList,
        dual_cabs: nn.ModuleList,
        normalisation_value: float = 1.0,
        use_checkpoint: bool = True,
    ):
        """Initialise the LPD algorithm.

        Args:
            n_iter (int): The number of iterations.
            projector (RegularPolygonPETProjector): The projector.
            primal_layers (nn.ModuleList): The primal layers.
            dual_layers (nn.ModuleList): The dual layers.
            primal_cabs (nn.ModuleList): The primal cross-attention blocks.
            dual_cabs (nn.ModuleList): The dual cross-attention blocks.
            normalisation_value (float, optional): The normalisation value. Defaults to 1.0.
        """
        super().__init__()

        self.n_iter = n_iter
        self.proj = projector
        self.dual_shape = self.proj.out_shape

        # Define the projector layer
        self.proj_layer = TorchProjector.apply

        # Store the processing layers
        self.primal_layers = primal_layers
        self.dual_layers = dual_layers
        self.primal_cabs = primal_cabs
        self.dual_cabs = dual_cabs

        # Normalisation value for the projector
        self.normalisation = normalisation_value

        # Flag to enable/disable checkpointing
        self.use_checkpoint = use_checkpoint

        # Rearrange layers to convert between 2D and 3D tensors in the Primal and Dual spaces
        self._to2DP = Rearrange("b c h s w -> (b s) c h w")
        self._to3DP = Rearrange("(b s) c h w -> b c h s w", s=self.dual_shape[-1])

        self._to2DD = Rearrange("b c h w s -> (b s) c h w")
        self._to3DD = Rearrange("(b s) c h w -> b c h w s", s=self.dual_shape[-1])

    def _checkpoint(self, func, *args):
        """Checkpointing utility.

        Args:
            func (callable): The function to checkpoint.
            *args: The arguments to pass to the function.

        Returns:
            torch.Tensor: The result of the checkpointed function.
        """
        if self.use_checkpoint:
            return cp.checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LPD algorithm.

        Args:
            x (torch.Tensor): The input sinogram.

        Returns:
            torch.Tensor: The reconstructed image.
        """

        # Initialise the primal and dual variables
        sino_step = self._to3DD(self.dual_layers[0](self._to2DD(x)))
        img_step = self._to3DP(
            self.primal_layers[0](
                self._to2DP(
                    self.proj_layer(sino_step.float(), self.proj, True)
                    / self.normalisation
                )
            )
        )

        # Create lists to concatenate the variables at each iteration
        sino_list = [x, sino_step]
        img_list = [img_step]

        # Run the LPD algorithm for the specified number of iterations
        for i in range(1, self.n_iter):

            # Dual concatenation and processing
            sino_list.append(self.proj_layer(img_step.float(), self.proj, False))
            sino_step_temp = self._to3DD(
                self.dual_layers[i](self._to2DD(torch.cat(sino_list, dim=1)))
            )

            # Dual Cross Attentive update
            sino_step = self._to3DD(
                self._checkpoint(
                    self.dual_cabs[i - 1],
                    [self._to2DD(sino_step_temp), self._to2DD(sino_step)],
                )
            )

            # Primal concatenation and processing
            img_list.append(
                self.proj_layer(sino_step.float(), self.proj, True) / self.normalisation
            )
            img_step_temp = self._to3DP(
                self.primal_layers[i](self._to2DP(torch.cat(img_list, dim=1)))
            )

            # Primal Cross Attentive update
            img_step = self._to3DP(
                self._checkpoint(
                    self.primal_cabs[i - 1],
                    [self._to2DP(img_step_temp), self._to2DP(img_step)],
                )
            )

            # Update the last elements of the lists
            sino_list[-1] = sino_step
            img_list[-1] = img_step

        return img_step.squeeze(1)
