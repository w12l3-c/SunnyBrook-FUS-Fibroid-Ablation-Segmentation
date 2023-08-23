# ===================================================================================================================
# File description:
# ------------------
# Trying to use MONAI
# ===================================================================================================================

# ======================= Imports =======================
import monai
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.networks.nets import vit, unetr
from monai.metrics import DiceMetric, confusion_matrix
from monai.losses import DiceCELoss 
from monai.inferers import sliding_window_inference
from monai.utils import first, set_determinism
from monai.visualize import plot_2d_or_3d_image
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
import pydicom
import numpy as np
import cv2

from spine import train_spine_dataset, val_spine_dataset, test_spine_dataset

# ======================= Hyperparameters =======================
set_determinism(seed=0)
torch.manual_seed(42)

NUM_WORKERS = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================= Transformation =======================
def load_dicom_as_numpy(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_array = dicom_data.pixel_array
    return dicom_array

dicom_loader = LoadImaged(keys=["img"], reader=lambda x: load_dicom_as_numpy(x))
png_loader = LoadImaged(keys=["seg"], reader=lambda x: np.asarray(Image.open(x)))


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

# train_ds = CacheDataset(data=train_spine_dataset, transform=train_transform, cache_rate=1.0, num_workers=12)
# train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=12)

# val_ds = CacheDataset(data=val_spine_dataset, transform=val_transform, cache_rate=1.0, num_workers=12)
# val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=12)

# Create the ViT model
ViT = vit.ViT(
    in_channels=3,
    img_size=320,
    patch_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    num_layers=12,
    pos_embed="conv",
    classification=True,
    num_classes=6,
    dropout_rate=0.1,
    spatial_dims=3,
    post_activation="Tanh",
)

UnetR = unetr.UNETR(
    in_channels=3,
    out_channels=7,
    img_size=320,
    feature_size= 16,
    hidden_size= 768,
    mlp_dim= 3072,
    num_heads= 12,
    pos_embed="conv",
    norm_name="instance",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0,
    spatial_dims=3,
)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(ViT.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

print(ViT)
print(UnetR)



