# =============================================================================
# File Description:
# ------------------
# This file is to contain the functions and architecture for the DeepLabV3 model
# =============================================================================

# =================== Imports =================== #
from transformers import AutoImageProcessor, BeitForSemanticSegmentation, TrainingArguments, Trainer
import evaluate
from datasets import Dataset, DatasetDict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

import torchvision
import torchvision.transforms as transforms

# ================== Pretrained Beit3 ================== #
class Beit3():
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        self.model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
            

# ====================== Dataset ====================== #            
# HuggingFace have a specific dataset structure
# Convert your dataset into a DatasetDict
def create_huggingface_dataset():
    # Convert your dataset into a DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data),
        "test": Dataset.from_dict(test_data),
    })

    # Print the dataset
    print(dataset_dict)
    
    
# ====================== Metrics ====================== #
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=1,
            ignore_index=255,
            reduce_labels=False,
        )

        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()

        return metrics 
    

def accuracy_iou(pred, target):
    pred_mask = pred > 0.5
    target_mask = target > 0.5

    intersection = torch.sum(pred_mask * target_mask)
    union = torch.sum(pred_mask + target_mask)

    iou = intersection / union
    return iou

def accuracy_intersect(pred, target):
    pred_mask = pred > 0.5
    target_mask = target > 0.5
    intersection = torch.sum(pred_mask * target_mask)
    pred_total = torch.sum(pred_mask == True)
    
    return intersection / pred_total

def accuracy_basic(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    correct = torch.sum(pred == target)
    return correct / pred.numel()


# ===================== Inference ====================== #
def display_seg(image, pred_seg, display=False):
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    palette = np.asarray([                           # Change palette later        
                            [0, 0, 0],
                            [120, 120, 120],
                            [180, 120, 120],
                            [6, 230, 230],
                            [80, 50, 50],
                            [4, 200, 3]
                    ])
    for label, color in enumerate(palette):
        color_seg[pred_seg == label, :] = color
        color_seg = color_seg[..., ::-1]  # BRG

        img = np.array(image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

    if display == True:
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.show()
    
    return color_seg


def inference(image_processor, model, images, display=False, save=False):
    for image in images:
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        # logits are of shape (batch_size, num_labels, height, width)
        logits = outputs.logits
        upsample = interpolate(
            logits,
            size=image.shape[::-1], # need to fix
            mode='bilinear',
            align_corners=False,
        )
        pred_seg = upsample.argmax(dim=1)[0]
        
        if save:
            display_seg(image, pred_seg, display)