from typing import Type

import torch
import torch.nn as nn


class UNet2D(nn.Module):
    """A 2D U-Net model architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        num_layers: int = 4,
        features_start: int = 32,
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        normalisation: Type[nn.Module] = nn.BatchNorm2d,
        dual: bool = False,
    ) -> None:
        """Initialise the 2D U-Net model.

        Args:
            in_channels (int, optional): The number of input channels. Defaults to 3.
            out_channels (int, optional): The number of output channels. Defaults to 1.
            num_layers (int, optional): The number of layers in the U-Net. Defaults to 4.
            features_start (int, optional): The number of features in the first layer. Defaults to 32.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            activation (nn.Module, optional): The activation function. Defaults to nn.ReLU.
            normalisation (nn.Module, optional): The normalisation layer. Defaults to nn.BatchNorm2d.
            dual (bool, optional): Whether the U-Net is used in the dual domain. Defaults to False.
        """
        super().__init__()

        self.num_layers = num_layers
        self.features_start = features_start
        self.kernel_size = kernel_size
        self.activation = activation
        self.normalization = normalisation

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else features_start * 2 ** (i - 1)
            out_channels_ = features_start * 2**i
            self.encoders.append(self._block(in_channels_, out_channels_))

        # Bottleneck
        in_channels_ = features_start * 2 ** (num_layers - 1)
        out_channels_ = features_start * 2**num_layers
        self.bottleneck = self._block(in_channels_, out_channels_)

        # Decoder and upsampler
        self.decoders = nn.ModuleList()
        self.upsampler = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_channels_ = features_start * 2**i * 2
            out_channels_ = features_start * 2**i
            # Select padding and output padding
            padding, output_padding = (
                ((1, 1), (1, 1)) if i == num_layers - 1 else ((0, 0), (0, 0))
            )
            if dual:
                padding, output_padding = (
                    ((0, 1), (0, 1))
                    if (num_layers - i - 1) % 2 == 0
                    else ((0, 0), (0, 0))
                )
            self.decoders.append(self._block(in_channels_, out_channels_))
            self.upsampler.append(
                self._upconv_block(in_channels_, out_channels_, padding, output_padding)
            )

        # Final layer
        self.final_conv = nn.Conv2d(
            features_start, out_channels, kernel_size=1, stride=1, padding=0
        )

    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a 2D convolutional block consisting of two convolutional layers,
            each followed by a normalization and activation layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: The convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            ),
            self.normalization(out_channels),
            self.activation(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            ),
            self.normalization(out_channels),
            self.activation(inplace=True),
        )

    def _upconv_block(
        self,
        in_channels: int,
        out_channels: int,
        padding: tuple,
        output_padding: tuple,
    ) -> nn.Sequential:
        """Creates an 2D upsampling block consisting of a transposed convolutional layer,
            each followed by a normalization and activation layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            padding (tuple): The padding to apply.
            output_padding (tuple): The output padding to apply.

        Returns:
            nn.Sequential: The upsampling block.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
            ),
            self.normalization(out_channels),
            self.activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D U-Net model.
            The input passes through the encoder path, bottleneck, and decoder path.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Encoder path
        encoder_list = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_list.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            x = self.upsampler[i](x)

            encoder_feature = encoder_list[-(i + 1)]
            x = torch.cat([x, encoder_feature], dim=1)
            x = decoder(x)

        # Final layer
        x = self.final_conv(x)
        return x


class UNet3D(nn.Module):
    """A 3D U-Net model architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        num_layers: int = 4,
        features_start: int = 32,
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        normalisation: Type[nn.Module] = nn.BatchNorm2d,
        dual: bool = False,
    ) -> None:
        """Initialise the 3D U-Net model.

        Args:
            in_channels (int, optional): The number of input channels. Defaults to 3.
            out_channels (int, optional): The number of output channels. Defaults to 1.
            num_layers (int, optional): The number of layers in the U-Net. Defaults to 4.
            features_start (int, optional): The number of features in the first layer. Defaults to 32.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            activation (nn.Module, optional): The activation function. Defaults to nn.ReLU.
            normalisation (nn.Module, optional): The normalisation layer. Defaults to nn.BatchNorm2d.
            dual (bool, optional): Whether the U-Net is used in the dual domain. Defaults to False.
        """
        super().__init__()

        self.num_layers = num_layers
        self.features_start = features_start
        self.kernel_size = kernel_size
        self.activation = activation
        self.normalization = normalisation

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else features_start * 2 ** (i - 1)
            out_channels_ = features_start * 2**i
            self.encoders.append(self._block(in_channels_, out_channels_))

        # Bottleneck
        in_channels_ = features_start * 2 ** (num_layers - 1)
        out_channels_ = features_start * 2**num_layers
        self.bottleneck = self._block(in_channels_, out_channels_)

        # Decoder and upsampler
        self.decoders = nn.ModuleList()
        self.upsampler = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_channels_ = features_start * 2**i * 2
            out_channels_ = features_start * 2**i
            # Select padding and output padding
            padding, output_padding = (
                ((1, 1, 1), (1, 1, 1))
                if i == num_layers - 1
                else ((0, 0, 0), (0, 0, 0))
            )
            if dual:
                padding, output_padding = (
                    ((0, 1, 1), (0, 1, 1))
                    if i == 2
                    else (((0, 0, 0), (0, 0, 0)) if i == 1 else ((0, 1, 0), (0, 1, 0)))
                )

            self.decoders.append(self._block(in_channels_, out_channels_))
            self.upsampler.append(
                self._upconv_block(in_channels_, out_channels_, padding, output_padding)
            )

        # Final layer
        self.final_conv = nn.Conv3d(
            features_start, out_channels, kernel_size=1, stride=1, padding=0
        )

    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a 3D convolutional block consisting of two convolutional layers,
            each followed by a normalization and activation layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: The convolutional block.
        """
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            ),
            self.normalization(out_channels),
            self.activation(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            ),
            self.normalization(out_channels),
            self.activation(inplace=True),
        )

    def _upconv_block(
        self,
        in_channels: int,
        out_channels: int,
        padding: tuple,
        output_padding: tuple,
    ) -> nn.Sequential:
        """Creates a 3D upsampling block consisting of a transposed convolutional layer
            each followed by a normalization and activation layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            padding (tuple): The padding to apply.
            output_padding (tuple): The output padding to apply.

        Returns:
            nn.Sequential: The upsampling block.
        """
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
            ),
            self.normalization(out_channels),
            self.activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3D U-Net model.
            The input passes through the encoder path, bottleneck, and decoder path.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Encoder path
        encoder_list = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_list.append(x)
            x = nn.MaxPool3d(kernel_size=2, stride=2)(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            x = self.upsampler[i](x)

            encoder_feature = encoder_list[-(i + 1)]

            x = torch.cat([x, encoder_feature], dim=1)
            x = decoder(x)

        # Final layer
        x = self.final_conv(x)
        return x
