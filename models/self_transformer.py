import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.functions.pos_encodings import LearnableEncoding, PositionalEncoding


class SelfTransformerBlock(nn.Module):
    """A basic Transformer block with multi-head self-attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        """Initialises the Transformer block (self attention).

        Args:
            embedding_dim (int): Total embedding dimension (heads * head_dim).
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            bias (bool, optional): Whether to include bias in the attention layer. Defaults to False.
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            batch_first=True,
            dropout=dropout,
        )

        # MLP block with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Layer normalizations and dropout
        self.norm_att = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_norm = self.norm_att(x)

        with torch.autocast(enabled=False, device_type=str(x.device)):

            att_out, _ = self.mha(
                query=x_norm.float(),
                key=x_norm.float(),
                value=x_norm.float(),
                need_weights=False,
            )

        x = x + att_out

        x_norm = self.norm_mlp(x)

        x = x + self.dropout(self.mlp(x_norm))

        return x


class SelfViT2D(nn.Module):
    """A Transformer-based block for 2D image processing that uses self-attention."""

    def __init__(
        self,
        in_channels,
        img_shape,
        patch_size: int = 7,
        embed_dim: int = 128,
        num_heads: int = 2,
        dual: bool = False,
        indices=None,
        learnable_pos: bool = False,
    ):
        """Initialise the Vision Transformer block.

        Args:
            in_channels (int): Number of input channels.
            img_shape (tuple): Shape of the input image (height, width, depth).
            patch_size (int, optional): Patch size for the image. Defaults to 7.
            embed_dim (int, optional): Embedding dimension. Defaults to 128.
            num_heads (int, optional): Number of attention heads. Defaults to 2.
            dual (bool, optional): Whether the U-Net is used in the dual domain. Defaults to False.
            indices (torch.Tensor): Indices used for positional encoding (dual domain).
            learnable_pos (bool, optional): Whether to use learnable positional encodings. Defaults to False.
        """
        super().__init__()

        # Store variables values
        self.dual = dual

        # Store the image shape
        self.im_h, self.im_w, self.im_d = img_shape

        if dual and indices is not None:
            self.register_buffer("indices", indices)

            # Input embedding
            self.embed = nn.Sequential(
                Rearrange("bd c t v -> bd t (c v)"),
                nn.Linear(in_channels * indices.shape[1], embed_dim),
            )

            # Positional encodings
            self.positional_encodings = (
                LearnableEncoding(in_channels * indices.shape[0], embed_dim)
                if learnable_pos
                else PositionalEncoding()
            )

            # Output projection
            self.unembed = nn.Sequential(
                nn.Linear(embed_dim, self.indices.shape[1]),
                Rearrange("bd t (c v) -> bd c t v", c=1),
            )
        elif not dual:

            # Input embedding
            self.embed = nn.Sequential(
                Rearrange(
                    "bd c (p1 h) (p2 w) -> bd (p1 p2) (c h w)",
                    h=patch_size,
                    w=patch_size,
                ),
                nn.Linear(in_channels * patch_size * patch_size, embed_dim),
            )

            # Positional encodings
            self.positional_encodings = LearnableEncoding(
                in_channels * (self.im_h // patch_size) * (self.im_d // patch_size),
                embed_dim,
            )

            # Output projection
            self.unembed = nn.Sequential(
                nn.Linear(embed_dim, patch_size * patch_size),
                Rearrange(
                    "bd (p1 p2) (h w c) -> bd c (p1 h) (p2 w)",
                    c=1,
                    h=patch_size,
                    p1=self.im_h // patch_size,
                ),
            )
        else:
            raise ValueError("Indices must be provided for the dual domain.")

        # Transformer block
        self.transformer = SelfTransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Image Transformer block.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        if self.dual:
            # Select the input slices based on indices
            x0 = x[:, :, self.indices[:, :, 0].long(), self.indices[:, :, 1].long()]
        else:
            x0 = x

        # Embed the selected input and add positional encodings
        Z = self.positional_encodings(self.embed(x0))

        # Pass through the Transformer block
        Z = self.transformer(Z)

        # Unembed and reshape the output back to the input shape
        unembedded_x = self.unembed(Z)

        if self.dual:
            x0 = torch.zeros_like(x).float()
            x0[:, :, self.indices[:, :, 0].long(), self.indices[:, :, 1].long()] = (
                unembedded_x.float() * self.indices[:, :, 2]
            )
            out = x + x0
        else:
            out = x + unembedded_x

        return out
