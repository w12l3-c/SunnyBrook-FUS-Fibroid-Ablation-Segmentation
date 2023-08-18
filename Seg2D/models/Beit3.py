# =============================================================================
# File Description:
# ------------------
# This file is to contain the functions and architecture for the Beit3 model
#
# Update: This computer cannot handle Beit3 model, 
#         the estimate gpu usage is 25GB, but this computer only has 8GB
# =============================================================================

# =================== Imports =================== #
from transformers import AutoImageProcessor, BeitForSemanticSegmentation, BeitFeatureExtractor, TrainingArguments, Trainer
import evaluate
from datasets import Dataset, DatasetDict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

import torchvision
import torchvision.transforms as transforms

# ================== Pretrained Beit3 ================== #
class Beit3():
    def __init__(self, id2label, label2id):
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        self.feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        self.model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640", id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
            

# ====================== Dataset ====================== #            
# HuggingFace have a specific dataset structure
def dataset2dict(dataset):
    data_dict = {}
    image_array = []
    mask_array = []
    for i in dataset:
        image_array.append(i['img'])
        mask_array.append(i['mask'])
    data_dict['image'] = image_array
    data_dict['mask'] = mask_array
    return data_dict
        
# Convert your dataset into a DatasetDict
def create_huggingface_dataset(train_dataset, val_dataset, test_dataset):
    train_dict = dataset2dict(train_dataset)
    val_dict = dataset2dict(val_dataset)
    test_dict = dataset2dict(test_dataset)
    
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    })
    
    return dataset_dict

def train_transforms(beit, image):
    image = beit.image_processor(image)
    return image

def val_transforms(beit, image):
    image = beit.image_processor(image)
    return image

    
# ====================== Metrics ====================== #
metric = evaluate.load("mean_iou")

def compute_metrics(eval_preds):
  metric = evaluate.load("glue", "mrpc")
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

def compute_metrics(pred):
    with torch.no_grad():
        logits, labels = pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2label),
                ignore_index=0,
            )
        
        return metrics


def accuracy_iou(pred, target):
    num_labels = target.unique().numel()
    ious = []
    
    for label in range(1, num_labels):
        pred_mask = pred == label
        target_mask = target == label
        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask + target_mask)

        iou = intersection / union
        ious.append(iou)

    mean_iou = torch.mean(torch.stack(ious))
    return mean_iou

def accuracy_intersect(pred, target):
    num_labels = target.unique().numel()
    intersects = []
    
    for label in range(1, num_labels):
        pred_mask = pred == label
        target_mask = target == label
        intersection = torch.sum(pred_mask * target_mask)
        score = intersection / torch.sum(target_mask)
        intersects.append(score)
    
    mean_intersect = torch.mean(torch.stack(intersects))
    
    return mean_intersect

def accuracy_dice(pred, target):
    num_labels = target.unique().numel()
    dices = []
    
    for label in range(1, num_labels):
        pred_mask = pred == label
        target_mask = target == label
        intersection = torch.sum(pred_mask * target_mask)
        dice = (2 * intersection) / (torch.sum(pred_mask) + torch.sum(target_mask))
        dices.append(dice)
    
    mean_dice = torch.mean(torch.stack(dices))
    
    return mean_dice

def accuracy_basic(pred, target):
    correct = torch.sum(pred == target)
    return correct / pred.numel()


# ===================== Inference ====================== #
