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
import datetime

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

# The training functions are literally the same as the pytorch ones
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter(f'./runs/ViT_Multi_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
for epoch in range(10):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{10}")
    ViT.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
        
        outputs = model(inputs)
        
        loss = loss_function(outputs, labels)
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Usually the val_interval = 1 so you don't have to do this
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                roi_size = (96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_segmentation2d_dict.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


