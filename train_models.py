import os
import pickle
import sys
from typing import Literal

import torch
from loguru import logger
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from configs import _models
from configs._scanners import get_miniPET_geometry
from generators.mixed import MixedDataset
from models.functions.embeddings import sinogram_patching
from models.functions.losses import MAELoss, MSEDiceLoss, MSELoss
from utils.inits import (
    EarlyStopper,
    get_vram,
    init_loguru,
    init_pytorch,
    init_weights,
    load_config,
)
from utils.telegram import send_message

# Check if the argument is provided
if len(sys.argv) < 2:
    print("Error: Configuration file path not provided.")
    print(f"Usage: python {os.path.basename(__file__)} <config_file_path>")
    sys.exit(1)

# Get the parameters for the training
params = load_config(sys.argv[1])

# General params
FILE_NAME = params["name"]
SAVE_OUTPUT = params["save_output"]

# Initialise the logger and get the output path
output_path, file_name = init_loguru(
    save_to_file=SAVE_OUTPUT,
    config_name=FILE_NAME,
)
# Initialise the device
device: torch.device = init_pytorch()

# Set volume dimensions
IMG_WIDTH: int = params["img_width"]
IMG_HEIGHT: int = params["img_height"]
IMG_DEPTH: int = params["img_depth"]

# Dataset parameters
BATCH_SIZE: int = params["batch_size"]
TRAIN_SAMPLES: int = params["train_samples"]

# Noise and blur parameters
NOISE_INTERVAL: tuple = params["noise_interval"]
KERNEL_SIZE: int = params["kernel_size"]
SIGMA: float = params["sigma"]

# LPD parameters
NUM_LPD_ITERATIONS: int = params["num_lpd_iterations"]

# Training parameters
WEIGHT_DECAY: float = params["weight_decay"]
LEARNING_RATE: float = params["learning_rate"]
USE_EARLY_STOPPING: bool = params["use_early_stopping"]

# Num of epochs
NUM_EPOCHS: int = params["num_epochs"]

# Dataset modality
MOD: Literal["ellipsoids", "shapes", "mixed"] = "mixed"
if "dataset" in params:
    MOD = params["dataset"]

# Create the miniPET geometry
miniPET_geometry = get_miniPET_geometry(
    device=device, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, num_rings=IMG_DEPTH
)

# Shapes parameters
NUM_LABELS: int = params["num_labels"]

# Create the synthetic dataset
synthetic_dataset = MixedDataset(
    device,
    projector=miniPET_geometry.proj,
    n_imgs=TRAIN_SAMPLES,
    n_labels=NUM_LABELS,
    noise_interval=NOISE_INTERVAL,
    kernel_size=KERNEL_SIZE,
    sigma=SIGMA,
    modality=MOD,
)

# Select the loss function
if params["loss"] == "MSEDice":
    loss_function = MSEDiceLoss(device, alpha=params["alpha"])
elif params["loss"] == "MSE":
    loss_function = MSELoss()
elif params["loss"] == "MAE":
    loss_function = MAELoss()
else:
    raise ValueError("Invalid loss function.")

# Create the data loader
data_loader = DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Get the model name
NAME: str = params["name"]

# Initialise the full model
if "lpdUnet2D" in NAME:
    logger.info("LPD_Unet2D model loaded successfully!")
    model = _models.get_LPD_Unet2D_model(
        device,
        projector=miniPET_geometry.proj,
        n_iter=NUM_LPD_ITERATIONS,
    )

elif "lpdUnet3D" in NAME:
    logger.info("LPD_Unet3D model loaded successfully!")
    model = _models.get_LPD_Unet3D_model(
        device,
        projector=miniPET_geometry.proj,
        n_iter=NUM_LPD_ITERATIONS,
    )

else:
    # Get the transformer parameters
    PATCH_SIZE: int = params["patch_size"]
    EMBED_DIM: int = params["embedding_dim"]
    NUM_HEADS: int = params["num_heads"]
    LEARNABLE_POS: bool = params["learnable_pos"]

    # Create the miniPET geometry
    miniPET_geometry_patch = get_miniPET_geometry(
        device=device,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        num_rings=1,
        distance_between_rings=0,
    )
    # Get the indices for positional tokenisation of the sinograms
    INDICES = sinogram_patching(
        device=device,
        projector=miniPET_geometry_patch.proj,
        patch_size=PATCH_SIZE,
    )
    if "lpdUnetTransformer2D" in NAME:
        logger.info("2D Unet Transformer LPD model loaded successfully!")
        model = _models.get_LPD_UnetTransformer2D_model(
            device,
            projector=miniPET_geometry.proj,
            n_iter=NUM_LPD_ITERATIONS,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            indices=INDICES,
        )
    elif "crossImageLpdUnet2D" in NAME:
        logger.info("Cross_Image_LPD_Unet2D model loaded successfully!")
        model = _models.get_Cross_Image_LPD_Unet2D_model(
            device,
            projector=miniPET_geometry.proj,
            n_iter=NUM_LPD_ITERATIONS,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            learnable_pos=LEARNABLE_POS,
            use_checkpoint=params["use_checkpoint"],
        )
    elif "crossSinogramLpdUnet2D" in NAME:
        logger.info("Cross_Sinogram_LPD_Unet2D model loaded successfully!")
        model = _models.get_Cross_Sinogram_LPD_Unet2D_model(
            device,
            projector=miniPET_geometry.proj,
            n_iter=NUM_LPD_ITERATIONS,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            indices=INDICES,
            learnable_pos=LEARNABLE_POS,
            use_checkpoint=params["use_checkpoint"],
        )
    elif "crossUpdateLpdUnet2D" in NAME:
        logger.info("Cross_Update_LPD_Unet2D model loaded successfully!")
        model = _models.get_Cross_Update_LPD_Unet2D_model(
            device,
            projector=miniPET_geometry.proj,
            n_iter=NUM_LPD_ITERATIONS,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            embed_dim_dual=params["embedding_dim_dual"],
            num_heads=NUM_HEADS,
            indices=INDICES,
            learnable_pos=LEARNABLE_POS,
            use_checkpoint=params["use_checkpoint"],
        )
    elif "crossConcatLpdUnet2D" in NAME:
        logger.info("Cross_Concat_LPD_Unet2D model loaded successfully!")
        model = _models.get_Cross_Concat_LPD_Unet2D_model(
            device,
            projector=miniPET_geometry.proj,
            n_iter=NUM_LPD_ITERATIONS,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            embed_dim_dual=params["embedding_dim_dual"],
            num_heads=NUM_HEADS,
            indices=INDICES,
            learnable_pos=LEARNABLE_POS,
            use_checkpoint=params["use_checkpoint"],
        )
    else:
        raise ValueError("Invalid model name.")

# Move the model to the device
model = model.to(device)

# Print the number of parameters
logger.info(
    f"{params['name']} - number of parameters: {sum(p.numel() for p in model.parameters()):,}"
)

# Initialise the weights of the model
model.apply(init_weights)

# Separate the parameters for weight decay and no weight decay
decay = []
no_decay = []

for name, param in model.named_parameters():
    if (
        "weight" in name
        and "batchnorm"
        and "layernorm"
        and "relative_bias_table" not in name
    ):
        decay.append(param)
    else:
        no_decay.append(param)

# Create the AdamW optimizer
optimizer = AdamW(
    [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0},
    ],
    lr=LEARNING_RATE,
)

# Create the learning rate scheduler
lr_scheduler = OneCycleLR(
    optimizer,
    max_lr=5 * LEARNING_RATE,
    steps_per_epoch=len(data_loader),
    epochs=NUM_EPOCHS,
)

# Create the early stopper
early_stopper = EarlyStopper(patience=20)
stopped: bool = False

# Initialise torchmetrics
psnr_metric = PeakSignalNoiseRatio().to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

psnr_epoch = []
ssim_epoch = []
loss_epoch = []

weights_path = f"{output_path}/{file_name}_weights.pt"

# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

scaler = GradScaler(growth_interval=100)

for epoch in range(NUM_EPOCHS):

    running_loss = 0.0
    psnr_batch = []
    ssim_batch = []

    model.train()  # Set the model to training mode

    for batch, data in enumerate(data_loader):
        logger.debug(
            f"{epoch + 1}/{NUM_EPOCHS} - batch {batch + 1}/{len(data_loader)} {get_vram()}"
        )
        sinos, *ground_truth = data

        with torch.autocast(device_type=str(device)):
            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            recon_imgs = model(sinos.unsqueeze(1))
            loss_value = loss_function(recon_imgs, *ground_truth)

        # Backward pass with scaling
        scaler.scale(loss_value).backward()

        # Gradient norm clipping
        max_norm = 5.0
        clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()

        # Optionally update the learning rate
        skip_lr_sched = scale > scaler.get_scale()
        if not skip_lr_sched:
            lr_scheduler.step()

        # Update the current loss
        running_loss += loss_value.item()

        # Calculate PSNR and SSIM
        imgs, *_ = ground_truth
        psnr_batch.append(psnr_metric(recon_imgs.detach(), imgs.detach()))
        ssim_batch.append(ssim_metric(recon_imgs.detach(), imgs.detach()))

        # Delete the variables
        del recon_imgs, ground_truth, imgs, loss_value

    running_loss /= len(data_loader)

    # Calculate the average PSNR and SSIM over the epoch
    psnr_value = torch.stack(psnr_batch).mean().item()
    ssim_value = torch.stack(ssim_batch).mean().item()

    logger.info(
        f"{epoch + 1}/{NUM_EPOCHS} - loss {running_loss:.2e}; PSNR {psnr_value:.3f}; SSIM {ssim_value:.3f}"
    )

    # Append the metrics to the lists
    psnr_epoch.append(psnr_value)
    ssim_epoch.append(ssim_value)
    loss_epoch.append(running_loss)

    if USE_EARLY_STOPPING and epoch > 100:
        if early_stopper.early_stop(running_loss, model, weights_path):
            logger.info("Early stopping activated!")
            stopped = True
            break

if SAVE_OUTPUT:
    if not USE_EARLY_STOPPING:
        # Save the weights of the model
        torch.save(model.state_dict(), weights_path)
    # Save metrics
    metrics = {
        "loss": loss_epoch,
        "psnr": psnr_epoch,
        "ssim": ssim_epoch,
    }
    with open(f"{output_path}/{file_name}_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    # Send a message to the Telegram chat
    send_message(f"Training of {file_name} is completed!")
