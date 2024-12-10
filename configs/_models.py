import array_api_compat.torch as xp
import torch
from parallelproj import RegularPolygonPETProjector
from torch import nn

from models.cross_transformer import CrossViT2D
from models.legacy.self_transformer import LegacySelfViT2D
from models.lpd import LPD
from models.lpd_cross_concat import CrossConcatLPD
from models.lpd_cross_image import CrossImageLPD
from models.lpd_cross_sino import CrossSinogramLPD
from models.lpd_cross_update import CrossUpdateLPD
from models.unet import UNet2D, UNet3D


def get_LPD_Unet2D_model(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    n_iter: int,
) -> torch.nn.Module:
    """The LPD model with 2D Unet layers in the primal and dual branches.

    Args:
        device (torch.device): The device to use.
        projector (RegularPolygonPETProjector): The projector.
        n_iter (int): The number of LPD iterations.

    Returns:
        torch.nn.Module: The model.
    """

    # Initialise the LPD model
    primal_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=i + 1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=False,
            )
            for i in range(n_iter)
        ]
    )
    dual_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=(i + 1 if i == 0 else i + 2),
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=True,
            )
            for i in range(n_iter)
        ]
    )

    # Calculate the normalisation value
    norm_value = (projector.norm(xp, str(device))) ** 2

    # Initialise the full model
    model = LPD(
        n_iter=n_iter,
        projector=projector,
        primal_layers=primal_layers,
        dual_layers=dual_layers,
        normalisation_value=norm_value,
    )

    return model


def get_LPD_UnetTransformer2D_model(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    n_iter: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    indices: torch.Tensor,
) -> torch.nn.Module:
    """The LPD model with 2D Unet layers in the primal branch and 2D ViT layers in the dual branch.

    Args:
        device (torch.device): The device to use.
        projector (RegularPolygonPETProjector): The projector.
        n_iter (int): The number of LPD iterations.
        patch_size (int): The patch size for the Transformer layer.
        embed_dim (int): The embed dimension of the Transformer layer.
        num_heads (int): The number of attention num_heads.
        indices (torch.Tensor): The indices for positional tokenisation. Defaults to None.

    Returns:
        torch.nn.Module: The model.
    """

    # Initialise the LPD model
    primal_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=i + 1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=False,
            )
            for i in range(n_iter)
        ]
    )
    dual_layers = nn.ModuleList(
        [
            LegacySelfViT2D(
                in_channels=(i + 1 if i == 0 else i + 2),
                img_shape=projector.out_shape,
                indices=indices,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dual=True,
            )
            for i in range(n_iter)
        ]
    )

    # Calculate the normalisation value
    norm_value = (projector.norm(xp, str(device))) ** 2

    # Initialise the full model
    model = LPD(
        n_iter=n_iter,
        projector=projector,
        primal_layers=primal_layers,
        dual_layers=dual_layers,
        normalisation_value=norm_value,
    )

    return model


def get_LPD_Unet3D_model(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    n_iter: int,
) -> torch.nn.Module:
    """The LPD model with 3D Unet layers in the primal and dual branches.

    Args:
        device (torch.device): The device to use.
        projector (RegularPolygonPETProjector): The projector.
        n_iter (int): The number of LPD iterations.

    Returns:
        torch.nn.Module: The model.
    """

    # Initialise the LPD model
    primal_layers = nn.ModuleList(
        [
            UNet3D(
                in_channels=i + 1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm3d,
                dual=False,
            )
            for i in range(n_iter)
        ]
    )

    dual_layers = nn.ModuleList(
        [
            UNet3D(
                in_channels=(i + 1 if i == 0 else i + 2),
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm3d,
                dual=True,
            )
            for i in range(n_iter)
        ]
    )

    # Calculate the normalisation value
    norm_value = (projector.norm(xp, str(device))) ** 2

    # Initialise the full model
    model = LPD(
        n_iter=n_iter,
        projector=projector,
        primal_layers=primal_layers,
        dual_layers=dual_layers,
        normalisation_value=norm_value,
        use_3d=True,
    )

    return model


def get_Cross_Sinogram_LPD_Unet2D_model(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    n_iter: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    indices: torch.Tensor | None = None,
    learnable_pos: bool = False,
    use_checkpoint: bool = False,
) -> torch.nn.Module:
    """The LPD model with 2D Unet layers in the primal and dual branches.
    Cross-attention blocks are used to further process the sinogram data.

    Args:
        device (torch.device): The device to use.
        projector (RegularPolygonPETProjector): The projector.
        n_iter (int): The number of LPD iterations.
        patch_size (int): The patch size for the Transformer layer.
        embed_dim (int): The embed dimension of the Transformer layer.
        num_heads (int): The number of attention num_heads.
        indices (torch.Tensor): The indices for positional tokenisation. Defaults to None.
        learnable_pos (bool): Whether to use learnable positional encoding. Defaults to False.
        use_checkpoint (bool): Whether to use checkpointing. Defaults to False.

    Returns:
        torch.nn.Module: The model.
    """

    # Initialise the LPD model
    primal_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=i + 1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=False,
            )
            for i in range(n_iter)
        ]
    )

    dual_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=(i + 1 if i == 0 else i + 2),
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=True,
            )
            for i in range(n_iter)
        ]
    )

    dual_cabs = nn.ModuleList(
        [
            CrossViT2D(
                in_channels=1,
                img_shape=projector.out_shape,
                num_inputs=2,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dual=True,
                indices=indices,
                learnable_pos=learnable_pos,
            )
            for _ in range(n_iter - 1)
        ]
    )

    # Calculate the normalisation value
    norm_value = (projector.norm(xp, str(device))) ** 2

    # Initialise the full model
    model = CrossSinogramLPD(
        n_iter=n_iter,
        projector=projector,
        primal_layers=primal_layers,
        dual_layers=dual_layers,
        dual_cabs=dual_cabs,
        normalisation_value=norm_value,
        use_checkpoint=use_checkpoint,
    )

    return model


def get_Cross_Image_LPD_Unet2D_model(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    n_iter: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    indices: torch.Tensor | None = None,
    learnable_pos: bool = False,
    use_checkpoint: bool = False,
) -> torch.nn.Module:
    """The LPD model with 2D Unet layers in the primal and dual branches.
    Cross-attention blocks are used to further process the image data.

    Args:
        device (torch.device): The device to use.
        projector (RegularPolygonPETProjector): The projector.
        n_iter (int): The number of LPD iterations.
        patch_size (int): The patch size for the Transformer layer.
        embed_dim (int): The embed dimension of the Transformer layer.
        num_heads (int): The number of attention num_heads.
        indices (torch.Tensor): The indices for positional tokenisation.
        learnable_pos (bool): Whether to use learnable positional encoding. Defaults to False.
        use_checkpoint (bool): Whether to use checkpointing. Defaults to False.

    Returns:
        torch.nn.Module: The model.
    """

    # Initialise the LPD model
    primal_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=i + 1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=False,
            )
            for i in range(n_iter)
        ]
    )

    dual_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=(i + 1 if i == 0 else i + 2),
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=True,
            )
            for i in range(n_iter)
        ]
    )

    primal_cabs = nn.ModuleList(
        [
            CrossViT2D(
                in_channels=1,
                img_shape=projector.in_shape,
                num_inputs=2,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dual=False,
                indices=indices,
                learnable_pos=learnable_pos,
            )
            for _ in range(n_iter - 1)
        ]
    )

    # Calculate the normalisation value
    norm_value = (projector.norm(xp, str(device))) ** 2

    # Initialise the full model
    model = CrossImageLPD(
        n_iter=n_iter,
        projector=projector,
        primal_layers=primal_layers,
        dual_layers=dual_layers,
        primal_cabs=primal_cabs,
        normalisation_value=norm_value,
        use_checkpoint=use_checkpoint,
    )

    return model


def get_Cross_Update_LPD_Unet2D_model(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    n_iter: int,
    patch_size: int,
    embed_dim: int,
    embed_dim_dual: int,
    num_heads: int,
    indices: torch.Tensor | None = None,
    learnable_pos: bool = False,
    use_checkpoint: bool = False,
) -> torch.nn.Module:
    """The LPD model with 2D Unet layers in the primal and dual branches.
    Cross-attention blocks are used to fuse the outputs of the primal and dual branches.

    Args:
        device (torch.device): The device to use.
        projector (RegularPolygonPETProjector): The projector.
        n_iter (int): The number of LPD iterations.
        patch_size (int): The patch size for the Transformer layer.
        embed_dim (int): The embed dimension of the Transformer layer for the primal branch.
        embed_dim_dual (int): The embed dimension of the Transformer layer for the dual branch.
        num_heads (int): The number of attention num_heads.
        indices (torch.Tensor): The indices for positional tokenisation. Defaults to None.
        learnable_pos (bool): Whether to use learnable positional encoding. Defaults to False.
        use_checkpoint (bool): Whether to use checkpointing. Defaults to False.

    Returns:
        torch.nn.Module: The model.
    """

    # Initialise the LPD model
    primal_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=i + 1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=False,
            )
            for i in range(n_iter)
        ]
    )

    dual_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=(i + 1 if i == 0 else i + 2),
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=True,
            )
            for i in range(n_iter)
        ]
    )

    primal_cabs = nn.ModuleList(
        [
            CrossViT2D(
                in_channels=1,
                img_shape=projector.in_shape,
                num_inputs=2,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dual=False,
                indices=indices,
                learnable_pos=learnable_pos,
            )
            for _ in range(n_iter - 1)
        ]
    )

    dual_cabs = nn.ModuleList(
        [
            CrossViT2D(
                in_channels=1,
                img_shape=projector.out_shape,
                num_inputs=2,
                patch_size=patch_size,
                embed_dim=embed_dim_dual,
                num_heads=num_heads,
                dual=True,
                indices=indices,
            )
            for _ in range(n_iter - 1)
        ]
    )

    # Calculate the normalisation value
    norm_value = (projector.norm(xp, str(device))) ** 2

    # Initialise the full model
    model = CrossUpdateLPD(
        n_iter=n_iter,
        projector=projector,
        primal_layers=primal_layers,
        dual_layers=dual_layers,
        primal_cabs=primal_cabs,
        dual_cabs=dual_cabs,
        normalisation_value=norm_value,
        use_checkpoint=use_checkpoint,
    )

    return model


def get_Cross_Concat_LPD_Unet2D_model(
    device: torch.device,
    projector: RegularPolygonPETProjector,
    n_iter: int,
    patch_size: int,
    embed_dim: int,
    embed_dim_dual: int,
    num_heads: int,
    indices: torch.Tensor | None = None,
    learnable_pos: bool = False,
    use_checkpoint: bool = False,
) -> torch.nn.Module:
    """The LPD model with 2D Unet layers in the primal and dual branches.
    Cross-attention blocks are used to fuse the inputs of the primal and dual branches.

    Args:
        device (torch.device): The device to use.
        projector (RegularPolygonPETProjector): The projector.
        n_iter (int): The number of LPD iterations.
        patch_size (int): The patch size for the Transformer layer.
        embed_dim (int): The embed dimension of the Transformer layer for the primal branch.
        embed_dim_dual (int): The embed dimension of the Transformer layer for the dual branch.
        num_heads (int): The number of attention num_heads.
        indices (torch.Tensor): The indices for positional tokenisation. Defaults to None.
        learnable_pos (bool): Whether to use learnable positional encoding. Defaults to False.
        use_checkpoint (bool): Whether to use checkpointing. Defaults to False.

    Returns:
        torch.nn.Module: The model.
    """

    # Initialise the LPD model
    primal_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=False,
            )
            for _ in range(n_iter)
        ]
    )

    dual_layers = nn.ModuleList(
        [
            UNet2D(
                in_channels=1,
                out_channels=1,
                num_layers=3,
                features_start=32,
                kernel_size=3,
                activation=nn.ReLU,
                normalisation=nn.BatchNorm2d,
                dual=True,
            )
            for _ in range(n_iter)
        ]
    )

    primal_cabs = nn.ModuleList(
        [
            CrossViT2D(
                in_channels=1,
                num_inputs=i + 2,
                img_shape=projector.in_shape,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dual=False,
                indices=indices,
                learnable_pos=learnable_pos,
                concat=True,
            )
            for i in range(n_iter - 1)
        ]
    )

    dual_cabs = nn.ModuleList(
        [
            CrossViT2D(
                in_channels=1,
                num_inputs=i + 3,
                img_shape=projector.out_shape,
                patch_size=patch_size,
                embed_dim=embed_dim_dual,
                num_heads=num_heads,
                dual=True,
                indices=indices,
                learnable_pos=learnable_pos,
                concat=True,
            )
            for i in range(n_iter - 1)
        ]
    )

    # Calculate the normalisation value
    norm_value = (projector.norm(xp, str(device))) ** 2

    # Initialise the full model
    model = CrossConcatLPD(
        n_iter=n_iter,
        projector=projector,
        primal_layers=primal_layers,
        dual_layers=dual_layers,
        primal_cabs=primal_cabs,
        dual_cabs=dual_cabs,
        normalisation_value=norm_value,
        use_checkpoint=use_checkpoint,
    )

    return model
