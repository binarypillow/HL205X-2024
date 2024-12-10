import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """A 2D Positional Encoding block."""

    def __init__(self) -> None:
        """Initialise the PositionalEncoding block."""
        super().__init__()

    def positional_encoding(self, sequence_length: int, embed_dim: int) -> torch.Tensor:
        """Generate the positional encoding based on sine and cosine functions.

        Args:
            sequence_length (int): The length of the sequence.
            embed_dim (int): The embedding dimension.

        Returns:
            torch.Tensor: The positional encoding.
        """
        pe = torch.zeros(sequence_length, embed_dim)
        position = torch.arange(0, sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PositionalEncoding block.

        Args:
            x (torch.Tensor): The input tensor (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: The output tensor with positional encoding added.
        """
        b, sequence_length, embed_dim = x.size()

        # Generate positional encoding
        pe = self.positional_encoding(sequence_length, embed_dim).to(x.device)

        # Add positional encoding to input
        x = x + pe
        return x


class LearnableEncoding(nn.Module):
    """A 2D Learnable Encoding block."""

    def __init__(self, sequence_length: int, embed_dim: int) -> None:
        """Initialise the LearnableEncoding block.

        Args:
            sequence_length (int): The length of the sequence.
            embed_dim (int): The embedding dimension.
        """

        super().__init__()
        self.positional_encodings = nn.Parameter(
            torch.randn(1, sequence_length, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LearnableEncoding block.

        Args:
            x (torch.Tensor): The input tensor (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: The output tensor with positional encoding added.
        """

        return x + self.positional_encodings
