import monai
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.networks.nets import vit, unetr
from monai.metrics import DiceMetric, confusion_matrix
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

from spine import train_spine_dataset, val_spine_dataset, test_spine_dataset

set_determinism(seed=0)

# 
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

# Create the ViT model
model = vit(model_config)

print(model)

