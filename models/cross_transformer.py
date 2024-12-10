import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.functions.pos_encodings import LearnableEncoding, PositionalEncoding


class CrossTransformerBlock(nn.Module):
    """A basic Transformer block with multi-head cross-attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        concat: bool = False,
    ):
        """Initialise the Transformer block (cross attention).

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            bias (bool, optional): Whether to include bias in the attention layer. Defaults to False.
            concat (bool, optional): Whether to concatenate the input tensors. Defaults to False.
        """
        super().__init__()

        self.concat = concat

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            batch_first=True,
            dropout=dropout,
        )
        # MLP blocks with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Layer normalizations and dropout
        self.norm_att_1 = nn.LayerNorm(embed_dim)
        self.norm_att_2 = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x1 (torch.Tensor): Input tensor from the first domain.
            x2 (torch.Tensor): Input tensor from the second domain.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        x_norm1 = self.norm_att_1(x_1)
        x_norm2 = self.norm_att_2(x_2)

        with torch.autocast(enabled=False, device_type=str(x_1.device)):
            if not self.concat:
                att_out, _ = self.mha(
                    query=x_norm2.float(),
                    key=x_norm1.float(),
                    value=x_norm1.float(),
                    need_weights=False,
                )
            else:
                att_out, _ = self.mha(
                    query=x_norm1.float(),
                    key=x_norm2.float(),
                    value=x_norm2.float(),
                    need_weights=False,
                )

        x_1 = x_1 + att_out

        x_norm1 = self.norm_mlp(x_1)

        x_1 = x_1 + self.dropout(self.mlp(x_norm1))

        return x_1


class CrossViT2D(nn.Module):
    """A Vision Transformer block for 2D image processing that uses cross-attention.
    This block is implemented in two ways:
        - non concat version: the cross-attention is computed by swapping the query vectors
            and it can handle only two inputs.
        - concat version: the cross-attention is computed by concatenating the value
            and the key vectors and it can handle multiple inputs.
    """

    def __init__(
        self,
        in_channels,
        num_inputs: int,
        img_shape: tuple,
        patch_size: int = 7,
        embed_dim: int = 128,
        num_heads: int = 2,
        dual: bool = False,
        indices=None,
        learnable_pos: bool = False,
        concat: bool = False,
    ):
        """Initialise the 2D Vision Transformer block.

        Args:
            in_channels (int): Number of input channels.
            num_inputs (int): Number of input tensors.
            img_shape (tuple): Shape of the input image (height, width, depth).
            patch_size (int, optional): Patch size for the image. Defaults to 7.
            embed_dim (int, optional): Embedding dimension. Defaults to 128.
            num_heads (int, optional): Number of attention heads. Defaults to 2.
            dual (bool, optional): Whether the U-Net is used in the dual domain. Defaults to False.
            indices (torch.Tensor): Indices used for positional encoding (dual domain).
            learnable_pos (bool, optional): Whether to use learnable positional encodings. Defaults to False.
            concat (bool, optional): Whether to concatenate the input tensors. Defaults to False.
        """
        super().__init__()

        # Store variables values
        self.dual = dual
        self.num_inputs = num_inputs
        self.concat = concat

        # Store the image shape
        self.im_h, self.im_w, self.im_d = img_shape

        if dual and indices is not None:
            self.register_buffer("indices", indices)

            # Input embedding
            self.embeds = nn.ModuleList(
                [
                    nn.Sequential(
                        Rearrange("bd c t v -> bd t (c v)"),
                        nn.Linear(in_channels * indices.shape[1], embed_dim),
                    )
                    for _ in range(num_inputs)
                ]
            )

            # Positional encodings
            self.pos_encodings = (
                nn.ModuleList(
                    [
                        LearnableEncoding(in_channels * indices.shape[0], embed_dim)
                        for _ in range(num_inputs)
                    ]
                )
                if learnable_pos
                else nn.ModuleList([PositionalEncoding() for _ in range(num_inputs)])
            )

            # Output projection
            self.unembeds = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(embed_dim, self.indices.shape[1]),
                        Rearrange("bd t (c v) -> bd c t v", c=1),
                    )
                    for _ in range(num_inputs)
                ]
            )
        elif not dual:

            # Input embedding
            self.embeds = nn.ModuleList(
                [
                    nn.Sequential(
                        Rearrange(
                            "bd c (p1 h) (p2 w) -> bd (p1 p2) (c h w)",
                            h=patch_size,
                            w=patch_size,
                        ),
                        nn.Linear(in_channels * patch_size * patch_size, embed_dim),
                    )
                    for _ in range(num_inputs)
                ]
            )

            # Positional encodings
            self.pos_encodings = (
                nn.ModuleList(
                    [
                        LearnableEncoding(
                            (self.im_h // patch_size) * (self.im_d // patch_size),
                            embed_dim,
                        )
                        for _ in range(num_inputs)
                    ]
                )
                if learnable_pos
                else nn.ModuleList([PositionalEncoding() for _ in range(num_inputs)])
            )

            # Output projection
            self.unembeds = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(embed_dim, patch_size * patch_size),
                        Rearrange(
                            "bd (p1 p2) (h w c) -> bd c (p1 h) (p2 w)",
                            c=1,
                            h=patch_size,
                            p1=self.im_h // patch_size,
                        ),
                    )
                    for _ in range(num_inputs)
                ]
            )
        else:
            raise ValueError("Indices must be provided for the dual domain.")

        # Transformer block
        self.transformer = CrossTransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads, concat=concat
        )

    def forward(self, x_list: list) -> torch.Tensor:
        """Forward pass through the 2D Image Transformer block.

        Args:
            x_list (list): List of input tensors.

        Returns:
            torch.Tensor: Output tensor after processing.
        """

        if self.num_inputs is not len(x_list):
            raise ValueError("Number of inputs must match the number of input tensors.")
        elif self.num_inputs < 2:
            raise ValueError(
                "At least two inputs are required for the cross-attention."
            )

        if self.dual:
            # Select the input slices based on indices
            x0_list = [
                x[:, :, self.indices[:, :, 0].long(), self.indices[:, :, 1].long()]
                for x in x_list
            ]
        else:
            x0_list = x_list

        # Embed the selected input and add positional encodings
        Z0_list = [
            pos_enc(embed(x0))
            for embed, x0, pos_enc in zip(self.embeds, x0_list, self.pos_encodings)
        ]

        Z_list = []
        if self.num_inputs == 2 and not self.concat:
            for i in range(len(Z0_list)):
                Z = self.transformer(Z0_list[i], Z0_list[1 - i])
                Z_list.append(Z)

        elif self.concat:
            for i in range(len(Z0_list)):
                current_Z = Z0_list[i]
                all_Z = torch.cat(Z0_list, dim=1)  # Concatenate all Z tensors
                Z = self.transformer(current_Z, all_Z)
                Z_list.append(Z)
        else:
            raise ValueError(
                "Number of inputs must be 2 for cross-attention without concatenation."
            )

        # Unembed and reshape the output back to the input shape
        unembedded_list = [unembed(Z) for unembed, Z in zip(self.unembeds, Z_list)]

        if self.dual:
            for i in range(len(unembedded_list)):
                x0 = torch.zeros_like(x_list[i]).float()
                x0[:, :, self.indices[:, :, 0].long(), self.indices[:, :, 1].long()] = (
                    unembedded_list[i].float() * self.indices[:, :, 2]
                )
                x_list[i] = x_list[i] + x0
        else:
            for i in range(len(unembedded_list)):
                x_list[i] = x_list[i] + unembedded_list[i]

        # Sum all the output tensors
        out = torch.zeros_like(x_list[0])
        for i in range(len(x_list)):
            out += x_list[i]

        return out
