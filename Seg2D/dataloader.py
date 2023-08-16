import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize

from patient import *

import cv2
import numpy as np
import pydicom
import datasets
from PIL import Image, ImageEnhance, ImageOps

# ---------------------- Data Augmentation ---------------------- #
colour_jitter = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
])

normalization = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229 , 0.224, 0.225])
])

def random_rotation(image, mask):
    # Random Rotation Angle
    angle = np.random.randint(-10, 11)

    # Apply the same rotation angle to both the image and the mask
    image_rotated = image.rotate(angle)
    mask_rotated = mask.rotate(angle)

    # Convert the rotated images back to NumPy arrays
    image_rotated = np.array(image_rotated)
    mask_rotated = np.array(mask_rotated)

    # Convert np arrays to PIL Image
    image_rotated_pil = Image.fromarray(image_rotated)
    mask_rotated_pil = Image.fromarray(mask_rotated)

    return image_rotated_pil, mask_rotated_pil

def gamma_correction_pil(image, gamma=1.0):
    enhancer = ImageEnhance.Brightness(image)
    gamma_corrected_image = enhancer.enhance(gamma)
    return gamma_corrected_image

def convert_to_PIL(pairs, img_size=(320, 320)):
    image_sizes = []
    pixel_spaces = []
    for pair in pairs:
        img_path = pair["img"]
        mask = pair["mask"]
        
        # The image format for different libraries:
            # cv2 image is (height, width, channels)
            # PIL image is (width, height) + mode
            # Tensor is (channels, height, width)
        
        file = pydicom.dcmread(img_path)
        img = file.pixel_array
        # pixel_spaceing = file.PixelSpacing    # Coronal doesn't have this attribute
        
        if img.shape not in image_sizes:
            image_sizes.append(img.shape)
        
        # if pixel_spaceing not in pixel_spaces:
        #     pixel_spaces.append(pixel_spaceing)
        
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.equalizeHist(img.astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(img)
        img = img.resize(img_size)
        img = img.convert("RGB")
        # img = ImageOps.equalize(img)
        # img = gamma_correction_pil(img, gamma=1.5)
        
        if mask is not None:
            mask = Image.open(mask)
            mask = mask.resize(img_size)
            mask = mask.convert("L")
        
        pair['img'] = img
        pair['mask'] = mask
    
    # Test out the pixel spacing and image sizes    
    # print(pixel_spaces)
    # print(image_sizes)
    return pairs

def convert_to_PIL_multi(pairs, color_map, img_size=(320, 320)):
    image_sizes = []
    for pair in pairs:
        img_path = pair["img"]
        mask = pair["mask"]
        
        # The image format for different libraries:
            # cv2 image is (height, width, channels)
            # PIL image is (width, height) + mode
            # Tensor is (channels, height, width)
        
        file = pydicom.dcmread(img_path)
        img = file.pixel_array
        
        if img.shape not in image_sizes:
            image_sizes.append(img.shape)
        
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.equalizeHist(img.astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(img)
        img = img.resize(img_size)
        img = img.convert("RGB")
        
        if mask is not None:
            mask = np.zeros((img_size[0], img_size[1], 3))
            for path in mask:
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, img_size)
                m_map = m > 10
                
                path_split = path.split("/")
                region = path_split[-2]
                color = color_map[region]
                
                mask[m_map] = color
            
            mask = Image.fromarray(mask.astype(np.uint8))
            
        pair['img'] = img
        pair['mask'] = mask
    
    return pairs

# ---------------------- Dataset & DataLoader ---------------------- #
class PatientDataset2D(torch.utils.data.Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform
    
    def __getitem__(self, index):
        img_mask_pair = self.pairs[index]
        img = img_mask_pair["img"]
        mask = img_mask_pair["mask"]
        
        if self.transform:
            img, mask = random_rotation(img, mask)
            
            img = colour_jitter(img)
            img = self.transform(img)
            img = normalization(img)
            
            if mask is not None:
                mask = self.transform(mask)
            
            return img, mask
        else:
            return img, mask
        
    def __len__(self):
        return len(self.pairs)


def create_dataloader(patient, transform, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=None):
    dataset = PatientDataset2D(patient, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate_fn)
    return dataloader


# ---------------------- Collate Functions ---------------------- #
# For same size images but keeping the ratio of the original image
def same_collate(batch):
    # Separate images and targets
    images, targets = zip(*batch)

    # Find the maximum height and width in the batch
    max_height = 320
    max_width = 320

    # Resize images and masks to the maximum dimensions
    resized_images = []
    resized_targets = []
    for img, target in zip(images, targets):
        # print(img.shape[1:], target.shape[1:])
        if img.shape[1:] != (max_height, max_width):
            # Square picture just resize
            if img.shape[1] % img.shape[2] == 0:
                resized_img = resize(img, size=(max_height, max_width))
                resized_images.append(resized_img)
            # Non-square picture pad (for rectangular mri)
            if img.shape[1:] == (320, 160):
                padded_img = torch.nn.functional.pad(img, (80, 80, 0, 0), mode='constant', value=0)
                resized_images.append(padded_img)
        else:
            resized_images.append(img)

        if target.shape[1:] != (max_height, max_width):
            # Square picture just resize
            if target.shape[1] % target.shape[2] == 0:
                resized_target = resize(target, size=(max_height, max_width))
                resized_targets.append(resized_target)
            # Non-square picture pad (for rectangular mri)
            if target.shape[1:] == (320, 160):
                padded_target = torch.nn.functional.pad(target, (80, 80, 0, 0), mode='constant', value=0)
                resized_targets.append(padded_target)
        else:
            resized_targets.append(target)

    # Convert resized images and targets to tensors and stack them
    images = torch.stack(resized_images)
    targets = torch.stack(resized_targets)

    return images, targets
    
# Collate  function for diffnerent size input images
def diff_collate(batch):
    images, targets = zip(*batch)
    return images, targets