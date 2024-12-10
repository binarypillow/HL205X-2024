import torch

from utils.geometry import PetGeometry


def get_miniPET_geometry(
    device: torch.device,
    img_width: int,
    img_height: int,
    num_rings: int = 35,
    distance_between_rings: float = 20,
) -> PetGeometry:
    """Create the miniPET geometry.

    Args:
        device (torch.device): The device to use.
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        num_rings (int, optional): The number of rings. Defaults to 35.
        distance_between_rings (float, optional): The distance between rings. Defaults to 20.

    Returns:
        PetGeometry: The miniPET geometry.
    """

    # MiniPet-3 Geometry parameters
    RADIUS: float = 211 / 2 - 2
    NUM_SIDES: int = 12
    NUM_LOR_ENDPOINTS_PER_SIDE: int = 35
    LOR_SPACING: float = 211 * torch.pi / (35 * 12)
    DISTANCE_BETWEEN_RINGS: float = distance_between_rings
    NUM_RINGS: int = num_rings
    SIMMETRY_AXIS: int = 1
    # LOR parameters
    RADIAL_TRIM: int = 155
    MAX_RING_DIFFERENCE: int = 0
    # Projection parameters
    IMG_SHAPE: tuple = (img_width, num_rings, img_height)
    VOXEL_SIZE: tuple = (80 / img_width, 40 / num_rings, 80 / img_height)

    # Create the miniPET geometry
    miniPET_geometry = PetGeometry(
        device=device,
        radius=RADIUS,
        num_sides=NUM_SIDES,
        num_lor_endpoints_per_side=NUM_LOR_ENDPOINTS_PER_SIDE,
        lor_spacing=LOR_SPACING,
        distance_between_rings=DISTANCE_BETWEEN_RINGS,
        num_rings=NUM_RINGS,
        simmetry_axis=SIMMETRY_AXIS,
        radial_trim=RADIAL_TRIM,
        max_ring_difference=MAX_RING_DIFFERENCE,
        img_shape=IMG_SHAPE,
        voxel_size=VOXEL_SIZE,
    )

    return miniPET_geometry
