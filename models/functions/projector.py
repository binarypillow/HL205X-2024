"""
    Adapted from https://github.com/gschramm/parallelproj
"""

import torch
from array_api_compat import device
from parallelproj import RegularPolygonPETProjector


class TorchProjector(torch.autograd.Function):
    """Custom PyTorch layer to use parallelproj projections with PyTorch's autograd engine."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        projector: RegularPolygonPETProjector,
        use_adjoint: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the autograd function.

        Args:
            ctx: The autograd context.
            x (torch.Tensor): The input tensor.
            operator (parallelproj.LinearOperator): The linear operator to apply.
            use_adjoint (bool, optional): Whether to use the adjoint operator. Defaults to False.

        Returns:
            torch.Tensor: The output tensor.
        """

        # Disable materialization of gradients to save memory
        ctx.set_materialize_grads(False)

        # Save the operator and the use_adjoint flag
        ctx.projector = projector
        ctx.use_adjoint = use_adjoint

        # Get the dimensions of the input tensor
        batch_size, channels, *_ = x.shape

        # Get the output shape of the operator
        output_shape = projector.in_shape if use_adjoint else projector.out_shape

        # Initialise the output tensor
        y = torch.zeros(
            (batch_size,) + (channels,) + output_shape,
            dtype=x.dtype,
            device=device(x),
        )

        # Apply the operator to each element of the input tensor
        for i in range(batch_size):
            for j in range(channels):
                if use_adjoint:
                    y[i, j, ...] = projector.adjoint(x[i, j, ...].detach())
                else:
                    y[i, j, ...] = projector(x[i, j, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass of the autograd function.

        Args:
            ctx: The autograd context.
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            tuple: The gradient of the input tensor.
        """

        # Get the operator and the use_adjoint flag from the context
        projector: RegularPolygonPETProjector = ctx.projector
        use_adjoint: bool = ctx.use_adjoint

        # Get the dimensions of the gradient tensor
        batch_size, channels, *_ = grad_output.shape

        # Get the input shape of the operator
        input_shape = projector.out_shape if use_adjoint else projector.in_shape

        # Initialise the gradient tensor
        x = torch.zeros(
            (batch_size,) + (channels,) + input_shape,
            dtype=grad_output.dtype,
            device=device(grad_output),
        )

        # Apply the adjoint operator to each element of the gradient tensor
        for i in range(batch_size):
            for j in range(channels):
                if use_adjoint:
                    x[i, j, ...] = projector(grad_output[i, j, ...].detach())
                else:
                    x[i, j, ...] = projector.adjoint(grad_output[i, j, ...].detach())

        return x, None, None
