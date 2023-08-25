# =============================================================================
# File Description:
# ------------------
# This file is to contain the functions and architecture for the Segformer model
#
# Have not debug yet
# =============================================================================

# =================== Imports =================== #
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments, Trainer
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

# ================== Constants ================== #
COLOR_DICT = {'Background':(256, 256, 256), 'Spine': (35, 132, 250), 'Bowel': (14, 240, 56), 'Muscle': (214, 51, 36), 'Skin': (240, 170, 31), 'hip_L': (173, 20, 250), 'hip_R': (131, 20, 250)}   # Color map in dictionart
COLOR_LIST = [v for v in COLOR_DICT.values()]   # Colour map in list
id2label = {i: k for i, (k, v) in enumerate(COLOR_DICT.items())}
label2id = {k: i for i, (k, v) in enumerate(COLOR_DICT.items())}
        
# ================== Pretrained Segformer ================== #
class Segformer():
    def __init__(self, path, id2label, label2id):
        self.image_processor = AutoImageProcessor.from_pretrained(path)
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(path, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
            
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

def train_transforms(model, image):
    image = model.image_processor(image, return_tensors="pt")
    return image

def val_transforms(model, image):
    image = model.image_processor(image, return_tensors="pt")
    return image

def test_transforms(model, image):
    image = model.image_processor(image, return_tensors="pt")
    return image

    
# ====================== Metrics ====================== #
metric = evaluate.load("mean_iou")
feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer_b3', do_reduce_labels=False)

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
    # currently using _compute instead of compute
    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
    metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=feature_extractor.do_reduce_labels,
        )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
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
