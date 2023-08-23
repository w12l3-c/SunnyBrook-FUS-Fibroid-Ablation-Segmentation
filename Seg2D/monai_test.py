# ===================================================================================================================
# File description:
# ------------------
# Trying to use MONAI
# ===================================================================================================================

# ======================= Imports =======================
import monai
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.networks.nets import vit, unetr, UNETR
from monai.metrics import DiceMetric, confusion_matrix
from monai.losses import DiceCELoss 
from monai.inferers import sliding_window_inference
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    Activations,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)

import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os

from spine import train_spine_dataset, val_spine_dataset, test_spine_dataset

# ======================= Hyperparameters =======================
set_determinism(seed=0)
torch.manual_seed(42)

NUM_WORKERS = os.cpu_count()

# ======================= Transformation =======================
train_transform = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg", spatial_size=[96, 96], pos=1, neg=1, num_samples=4
        ),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
    ]  
)

val_transform = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
]
)

model_config = {
    "img_size": 224,          # Input image size
    "patch_size": 16,         # Patch size
    "in_channels": 3,         # Number of input channels (e.g., 3 for RGB)
    "num_classes": 2,         # Number of output classes
    "hidden_dim": 768,        # Hidden dimension
    "mlp_dim": 3072,          # MLP dimension
    "num_heads": 12,          # Number of attention heads
    "num_layers": 12,         # Number of layers
    "channels": 3,            # Number of channels
    "dim": 256,               # Dimension
    "depth": 6,               # Depth
    "heads": 8,               # Number of heads
    "mlp_dim": 2048,          # MLP dimension
}

train_ds = CacheDataset(data=train_spine_dataset, transform=train_transform, cache_rate=1.0, num_workers=12)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=12)

val_ds = CacheDataset(data=val_spine_dataset, transform=val_transform, cache_rate=1.0, num_workers=12)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=12)

# Create the ViT model
model = vit(model_config)



