import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MSELoss(nn.Module):
    """Mean Squared Error loss function."""

    def __init__(self):
        """Initialise the Mean Squared Error loss function."""
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self,
        recon_imgs: torch.Tensor,
        imgs: torch.Tensor,
        label_maps: torch.Tensor,
        mappings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Mean Squared Error loss.

        Args:
            recon_imgs (torch.Tensor): The reconstructed images (batch of 3D images).
            imgs (torch.Tensor): The ground truth images (batch of 3D images).

        Returns:
            torch.Tensor: The global Mean Squared Error loss.
        """
        return self.loss(recon_imgs, imgs)


class MAELoss(nn.Module):
    """Mean Absolute Error loss function."""

    def __init__(self):
        """Initialise the Mean Absolute Error loss function."""
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(
        self,
        recon_imgs: torch.Tensor,
        imgs: torch.Tensor,
        label_maps: torch.Tensor,
        mappings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Mean Absolute Error loss.

        Args:
            recon_imgs (torch.Tensor): The reconstructed images (batch of 3D images).
            imgs (torch.Tensor): The ground truth images (batch of 3D images).

        Returns:
            torch.Tensor: The global Mean Absolute Error loss.
        """
        return self.loss(recon_imgs, imgs)


class MSEDiceLoss(nn.Module):
    """Combined MSE and Dice loss function."""

    def __init__(self, device: torch.device, alpha: float = 0.5):
        """
        Initialise the combined MSE and Dice loss function.

        Args:
            device (torch.device): The device to use.
            alpha (float): The weight of the MSE loss. Default is 0.5.
        """
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.MSE_loss = nn.MSELoss()
        self.Dice_loss = Dice()

    def _get_recon_labels(
        self, recon_imgs: torch.Tensor, mappings: torch.Tensor
    ) -> torch.Tensor:
        """This function reconstructs the labels map from the reconstructed images.

        Args:
            recon_imgs (torch.Tensor): The reconstructed images (batch of 3D images).
            mappings (torch.Tensor): The mapping between the image values and the original
                labels (batch of 1D mappings).

        Returns:
            torch.Tensor: The reconstructed label maps.
        """
        # Expand values tensor to match the shape of entries for broadcasting
        values_expanded = mappings.view(mappings.shape[0], 1, 1, 1, mappings.shape[-1])

        # Compute the absolute differences between entries and values
        differences = torch.abs(recon_imgs.unsqueeze(-1) - values_expanded)

        # Find the indices of the minimum values along the last dimension
        closest_indices = torch.argmin(differences, dim=-1)

        # One-hot encode the indices of the nearest key
        recon_label_maps = F.one_hot(
            closest_indices, num_classes=mappings.shape[-1]
        ).to(self.device)

        # Rearrange the dimensions to match the target label map shape
        recon_label_maps = rearrange(recon_label_maps, "b h w d c -> b c h w d")

        return recon_label_maps

    def forward(
        self,
        recon_imgs: torch.Tensor,
        imgs: torch.Tensor,
        label_maps: torch.Tensor,
        mappings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined MSE and Dice Score loss.

        Args:
            recon_imgs (torch.Tensor): The reconstructed images (batch of 3D images).
            imgs (torch.Tensor): The ground truth images (batch of 3D images).
            label_maps (torch.Tensor): The labels map (batch of 3D labels).
            mappings (torch.Tensor): The mapping between the one-hot encoded labels and the original labels
                (batch of 1D arrays).

        Returns:
            torch.Tensor: The combined global MSE and Dice Score loss.
        """
        # Compute the global MSE loss and scale it to be comparable to the Dice loss
        mse = self.MSE_loss(recon_imgs, imgs)

        recon_label_maps = self._get_recon_labels(recon_imgs, mappings)

        # Compute the Dice Score loss
        dice_score = self.Dice_loss(recon_label_maps, label_maps)

        # Return the weighted combination of the MSE and Dice score losses
        return self.alpha * mse * 100 + (1 - self.alpha) * dice_score


class Dice(nn.Module):
    """Dice Score loss function for one-hot encoded labels."""

    def __init__(self, epsilon: float = 1e-6):
        """Initialise the Dice Score loss function.

        Args:
            epsilon (float): Smoothing factor to avoid division by zero. Default is 1e-6.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, recon_label_maps: torch.Tensor, label_maps: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Dice Score loss for one-hot encoded labels.

        Args:
            recon_label_maps (torch.Tensor): The reconstructed label maps of shape (batch_size, n_classes, H, W, D).
            label_maps (torch.Tensor): The ground truth label maps in one-hot format, of shape (batch_size, n_classes, H, W, D).

        Returns:
            torch.Tensor: The Dice Score loss averaged across all batches and classes.
        """
        # Ensure both inputs are float tensors
        recon_label_maps = recon_label_maps.float()
        label_maps = label_maps.float()

        # Compute the intersection between the reconstructed and target label maps for each class and batch
        intersection = torch.sum(recon_label_maps * label_maps, dim=(2, 3, 4))

        # Compute the sum of the reconstructed and target label maps for each class and batch
        recon_sum = torch.sum(recon_label_maps, dim=(2, 3, 4))
        target_sum = torch.sum(label_maps, dim=(2, 3, 4))

        # Compute the Dice score for each class and batch
        dice_score = (2 * intersection) / (recon_sum + target_sum + self.epsilon)

        # Apply mask to only include non-empty label maps
        label_map_sum = torch.sum(label_maps, dim=(2, 3, 4))
        non_empty_mask = label_map_sum > 0
        dice_score = dice_score[non_empty_mask]

        # If all label maps are zero, return zero loss
        if dice_score.numel() == 0:
            return torch.tensor(0.0, device=recon_label_maps.device)

        return 1 - torch.mean(dice_score)
