import os
import pydicom
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------- Helper Functions ---------------------- #
# Flatten the mask for calculating dice loss
def flatten_mask(mask) -> torch.Tensor:
    flatten = mask.permute(1, 2, 3, 4, 0).contiguous()
    flatten = flatten.view(flatten.numel())
    flatten = F.one_hot(flatten)
    return flatten

# 3D volumes Interpolation
def transform(img_stack, mask_stack, d, w, h):
    assert d%16 == 0 and w%16 == 0 and h%16 == 0, "d, w, h must be divisible by 16"
    img_stack = F.interpolate(torch.from_numpy((img_stack).astype(np.float32)), size=(d, w, h), mode='trilinear', align_corners=False)
    mask_stack = F.interpolate(torch.from_numpy((mask_stack).astype(np.float32)), size=(d, w, h), mode='trilinear', align_corners=False)
    return img_stack, mask_stack

# Stack the dicom images into 3D array
def stack3d(img_list, mask_list):
    img_value_list = [pydicom.dcmread(img).pixel_array for img in img_list]
    img_stack = np.stack(img_value_list, axis=0)
    img_stack = np.expand_dims(img_stack, axis=0)
        
    mask_value_list = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) for mask in mask_list]
    mask_stack = np.stack(mask_value_list, axis=0)
    mask_stack = np.expand_dims(mask_stack, axis=0)
    
    return img_stack, mask_stack
