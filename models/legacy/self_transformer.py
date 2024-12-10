import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module with scaling."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialise the Multi-Head Self-Attention.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.head_dim = embed_dim // num_heads  # Dimension of each attention head
        self.scale = self.head_dim**-0.5  # Scaling factor for the dot-product attention
        self.dropout = nn.Dropout(dropout)

        # Linear layers for Query, Key, and Value projections
        self.qkv_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim, bias=False),
                    Rearrange("b n (h d) -> b h n d", h=num_heads),
                )
                for _ in range(3)
            ]
        )

        # Rearrange to original shape after attention
        self.rearrange = Rearrange("b h n d -> b n (h d)")
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Multi-Head Self-Attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        Q, K, V = [proj(x) for proj in self.qkv_proj]

        # Note: mixed precision does not work well with attention computations
        with torch.autocast(enabled=False, device_type="cuda"):
            # Compute scaled dot-product attention
            attention_scores = (
                torch.matmul(Q.float(), K.float().transpose(-1, -2)) * self.scale
            )
            attention_probs = self.softmax(attention_scores)
            attention_probs = torch.matmul(attention_probs, V.float())
            attention_out = self.dropout(attention_probs)

        return self.rearrange(attention_out)


class TransformerBlock(nn.Module):
    """A basic Transformer block with multi-head self-attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """Initialise the Transformer block.

        Args:
            embedding_dim (int): Total embedding dimension (heads * head_dim).
            heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
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
        # Multi-head attention with residual connection
        att_out = self.attention(x)

        x = self.norm_att(x + att_out)

        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = self.norm_mlp(x + self.dropout(mlp_out))

        return x


class LegacySelfViT2D(nn.Module):
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
        """
        super().__init__()

        # Store the dual flag
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
            # Learnable positional encodings
            self.positional_encodings = nn.Parameter(
                torch.randn(1, indices.shape[0], embed_dim)
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

            # Learnable positional encodings
            self.positional_encodings = nn.Parameter(
                torch.randn(
                    1,
                    (self.im_h // patch_size) * (self.im_d // patch_size),
                    embed_dim,
                )
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
        self.transformer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)

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
        Z = self.embed(x0) + self.positional_encodings

        # Pass through the Transformer block
        Z = self.transformer(Z)

        # Unembed and reshape the output back to the input shape
        unembedded_x = self.unembed(Z)

        # Expand the unembedded tensor to the full image size
        x_out = x[:, -1, :, :].unsqueeze(1)

        if self.dual:
            # Initialise the output tensor and update it with the processed values
            x0 = torch.zeros_like(x_out).float()
            x0[:, :, self.indices[:, :, 0].long(), self.indices[:, :, 1].long()] = (
                unembedded_x.float() * self.indices[:, :, 2]
            )
            out = x_out + x0
        else:
            out = x_out + unembedded_x

        return out
