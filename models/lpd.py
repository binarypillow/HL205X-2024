import torch
from einops.layers.torch import Rearrange
from parallelproj import RegularPolygonPETProjector
from torch import nn

from models.functions.projector import TorchProjector


class LPD(nn.Module):
    """The Learned Primal-Dual (LPD) algorithm."""

    def __init__(
        self,
        n_iter: int,
        projector: RegularPolygonPETProjector,
        primal_layers: nn.ModuleList,
        dual_layers: nn.ModuleList,
        normalisation_value: float = 1.0,
        use_3d: bool = False,
    ):
        """Initialise the LPD algorithm.

        Args:
            n_iter (int): The number of iterations.
            projector (RegularPolygonPETProjector): The projector.
            primal_layers (nn.ModuleList): The primal layers.
            dual_layers (nn.ModuleList): The dual layers.
            normalisation_value (float, optional): The normalisation value. Defaults to 1.0.
            use_3d (bool, optional): Whether to use 3D tensors. Defaults to False.
        """
        super().__init__()

        # Store the use_3d flag
        self.use_3d = use_3d

        # Store the number of iterations and the projector
        self.n_iter = n_iter
        self.proj = projector
        self.dual_shape = self.proj.out_shape

        # Define the projector layer
        self.proj_layer = TorchProjector.apply

        # Store the processing layers
        self.primal_layers = primal_layers
        self.dual_layers = dual_layers

        # Normalisation value for the projector
        self.normalisation = normalisation_value

        # Rearrange layers to convert between 2D and 3D tensors in the Primal and Dual spaces
        self._to2DP = Rearrange("b c h s w -> (b s) c h w")
        self._to3DP = Rearrange("(b s) c h w -> b c h s w", s=self.dual_shape[-1])

        self._to2DD = Rearrange("b c h w s -> (b s) c h w")
        self._to3DD = Rearrange("(b s) c h w -> b c h w s", s=self.dual_shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LPD algorithm.

        Args:
            x (torch.Tensor): The input sinogram.

        Returns:
            torch.Tensor: The reconstructed image.
        """

        # Initialise the primal and dual variables
        sino_step = (
            self._to3DD(self.dual_layers[0](self._to2DD(x)))
            if not self.use_3d
            else self.dual_layers[0](x)
        )
        img_step = (
            self._to3DP(
                self.primal_layers[0](
                    self._to2DP(
                        self.proj_layer(sino_step.float(), self.proj, True)
                        / self.normalisation
                    )
                )
            )
            if not self.use_3d
            else self.primal_layers[0](
                self.proj_layer(sino_step.float(), self.proj, True) / self.normalisation
            )
        )

        # Create lists to concatenate the variables at each iteration
        sino_list = [x, sino_step]
        img_list = [img_step]

        # Run the LPD algorithm for the specified number of iterations
        for i in range(1, self.n_iter):

            # Dual concatenation and update
            sino_list.append(self.proj_layer(img_step.float(), self.proj, False))
            sino_step += (
                self._to3DD(
                    self.dual_layers[i](self._to2DD(torch.cat(sino_list, dim=1)))
                )
                if not self.use_3d
                else self.dual_layers[i](torch.cat(sino_list, dim=1))
            )

            # Primal concatenation and update
            img_list.append(
                self.proj_layer(sino_step.float(), self.proj, True) / self.normalisation
            )
            img_step += (
                self._to3DP(
                    self.primal_layers[i](self._to2DP(torch.cat(img_list, dim=1)))
                )
                if not self.use_3d
                else self.primal_layers[i](torch.cat(img_list, dim=1))
            )

            # Update the last elements of the lists
            sino_list[-1] = sino_step
            img_list[-1] = img_step

        return img_step.squeeze(1)
