# =============================================================================
# File Description:
# ------------------
# This file is to contain the functions and architecture for the DeepLabV3 model
# =============================================================================

# =================== Imports =================== #
import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

# ================== Pretrained DeepLabV3 ================== #
def DeepLabV3(in_channels, num_classes, size=None):
    """
    Create a pretrained DeepLabV3 model for semantic segmentation with specified settings.
    Include model, data transformation, loss function, optimizer, and scheduler.

    Args:
        in_channels (int): Number of input channels for the model.
        num_classes (int): Number of classes for segmentation (background + objects).
        size (str or None): Size option for the model architecture. Available options are:
                            - 'heavy': DeepLabV3 with ResNet-101 backbone (largest and most accurate model)
                            - 'regular' or None: DeepLabV3 with ResNet-50 backbone (default)
                            - 'light': DeepLabV3 with MobileNet V3 Large backbone (smaller and faster model)

    Returns:
        model (torch.nn.Module): DeepLabV3 model with the specified architecture.
        transform (torchvision.transforms.Compose): Data transformation to be applied on input images.
        loss_fn (callable): Loss function used for training the model.
        optimizer (torch.optim.Optimizer): Optimizer used for training the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for the optimizer.

    Note:
        This function uses the PyTorch Hub to load pre-trained DeepLabV3 models from 'torchvision'.
        The returned model has the input and output layers modified to fit the specified number of channels.
        The loss function is automatically chosen based on the number of classes (binary or multilabel).

    Example:
        model, transform, loss_fn, optimizer, scheduler = DeepLabV3(in_channels=3, num_classes=21, size='regular')
    """
    if size == 'heavy':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    if size == None or size == 'regular':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    if size == 'light':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
    ])
    
    # Freeze all layers 
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the input layer and the output layer to fit our data
    model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    # Modify the forward function in the original model
    # def forward(self, x):
    #   x = self.backbone(x)
    #   main_logits = self.classifier(x['out'])
    #   aux_logits = self.aux_classifier(x['out'])
    #   return main_logits, aux_logits
    
    # model.forward = forward   # Overwrite forward methof of model
  
    # Loss function
    if num_classes == 1:
        loss_fn = smp.losses.DiceLoss('binary', from_logits=True)
        # loss_fn = smp.losses.JaccardLoss('binary', from_logits=True)
    if num_classes > 1:
        loss_fn = smp.losses.DiceLoss('multilabel', from_logits=True)
        # loss_fn = smp.losses.JaccardLoss('multilabel', from_logits=True)
        
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    
    return model, transform, loss_fn, optimizer, scheduler


# ================== Accuracy ================== #
def binary_segmentation_accuracy(logits, mask):
    """
    Compute accuracy for binary segmentation.

    Args:
        logits (torch.Tensor): Tensor containing the model's output logits.
                               Shape should be (batch_size, 1, height, width).
        mask (torch.Tensor): Ground truth mask tensor.
                             Shape should be (batch_size, 1, height, width).

    Returns:
        float: Accuracy value in the range [0, 1].
    """
    # Convert logits to binary predictions (0 or 1) using a threshold of 0.5
    predictions = (logits > 0.5).long()

    # Compute element-wise equality between predictions and mask
    correct_predictions = (predictions == mask).float()

    # Compute the overall accuracy (average of correct predictions)
    accuracy = correct_predictions.mean()

    return accuracy.item()


def binary_segmentation_iou(logits, mask):
    """
    Compute Intersection over Union (IOU) for binary segmentation.

    Args:
        logits (torch.Tensor): Tensor containing the model's output logits.
                               Shape should be (batch_size, 1, height, width).
        mask (torch.Tensor): Ground truth mask tensor.
                             Shape should be (batch_size, 1, height, width).

    Returns:
        float: IOU value in the range [0, 1].
    """
    # Convert logits to binary predictions (0 or 1) using argmax
    predictions = (logits > 0.5).long()

    # Calculate intersection and union areas
    intersection = torch.sum((predictions * mask).float())
    union = torch.sum(((predictions + mask) > 0).float())

    # Calculate IOU (Jaccard Index)
    iou = intersection / (union + 1e-7)  # Adding a small epsilon to avoid division by zero

    return iou.item()


# ================== Inference ================== #
def display(img, pred):
    """
    Overlay the segmentation prediction on the input image and display it.

    Args:
        img (PIL.Image.Image): Input image.
        pred (torch.Tensor): Segmentation prediction tensor.

    Returns:
        PIL.Image.Image: Overlay image with segmentation prediction.
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.randn(1, ) * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(pred.byte().cpu().numpy()).resize(img.size)
    r.putpalette(colors)

    plt.imshow(img, alpha=0.9)
    plt.imshow(r, alpha=0.5)
    
    return r
    

def inference(model, img_path, device):
    """
    Perform image segmentation inference using a model.

    Args:
        model (torch.nn.Module): Segmentation model.
        img_path (str): Path to the input image.
        device (torch.device): Device for inference (e.g., 'cuda' or 'cpu').

    Returns:
        PIL.Image.Image: Overlay image with segmentation prediction.
    """
    model = model.to(device)
    model.eval()
    
    img = Image.open(img_path)
    img = img.convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(img)
    img_batch = img_tensor.unsqueeze(0)
    img_batch = img_batch.to(device)
    
    with torch.no_grad():
        output_logits = model(img_batch)['out'][0]
    output_pred = torch.argmax(output_logits, dim=0)
    
    result = display(img, output_pred)
    return result
    
    
def inference_all(model, dir_path, device):
    """
    Perform image segmentation inference on all images in a directory.

    Args:
        model (torch.nn.Module): Segmentation model.
        dir_path (str): Path to the directory containing images.
        device (torch.device): Device for inference (e.g., 'cuda' or 'cpu').

    Returns:
        List[PIL.Image.Image]: List of overlay images with segmentation predictions.
    """
    dir_path = sorted(os.listdir(dir_path))
    slices = []
    for img_path in dir_path:
        silce = inference(model, img_path, device)
        slices.append(silce)   
    return slices