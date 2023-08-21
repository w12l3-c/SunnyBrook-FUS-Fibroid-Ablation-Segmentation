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

# ===================== Data Augmentation ===================== #
# Colour jitter - random brightness and contrast
colour_jitter = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
])

# Normalization for pretrained models
normalization = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229 , 0.224, 0.225])
])

def random_rotation(image, mask):
    """
    Randomly rotate an image and its corresponding mask.

    Args:
        image (PIL.Image): The input image.
        mask (PIL.Image): The corresponding mask image.

    Returns:
        PIL.Image: The rotated image.
        PIL.Image: The rotated mask.
    """
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
    """
    Apply gamma correction to an image using the PIL library.

    Args:
        image (PIL.Image): The input image.
        gamma (float, optional): The gamma correction factor. Default is 1.0.

    Returns:
        PIL.Image: The gamma-corrected image.
    """
    # Create Imgae Enhancer
    enhancer = ImageEnhance.Brightness(image)   
    # Enhance the brightness of the image
    gamma_corrected_image = enhancer.enhance(gamma)
    return gamma_corrected_image

def convert_to_PIL(pairs, img_size=(320, 320)):
    """
    Convert a list of image-mask pairs to PIL format.

    Args:
        pairs (list): A list of dictionaries containing image and mask paths.
        img_size (tuple, optional): The size to which images and masks should be resized. Default is (320, 320).

    Returns:
        list: A list of dictionaries with PIL images.
    """
    image_sizes = []    
    pixel_spaces = []  
    
    # Loop through all pairs in the dataset 
    for pair in pairs:
        # Get the image and mask paths
        img_path = pair["img"]
        mask = pair["mask"]
        
        # The image format for different libraries:
            # cv2 image is (height, width, channels)
            # PIL image is (width, height) + mode
            # Tensor is (channels, height, width)
        
        # Open the image as a dicom file and grab its pixel values
        file = pydicom.dcmread(img_path)
        img = file.pixel_array
        # pixel_spaceing = file.PixelSpacing    # Coronal doesn't have this attribute
        
        if img.shape not in image_sizes:
            image_sizes.append(img.shape)
        
        # if pixel_spaceing not in pixel_spaces:
        #     pixel_spaces.append(pixel_spaceing)
        
        # Data Augmentation and Preprocessing for image
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.equalizeHist(img.astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(img)
        img = img.resize(img_size)
        img = img.convert("RGB")
        # img = ImageOps.equalize(img)
        # img = gamma_correction_pil(img, gamma=1.5)
        
        # Data Augmentation and Preprocessing for mask
        if mask is not None:
            mask = Image.open(mask)
            mask = mask.resize(img_size)
            mask = mask.convert("L")
        
        # Update the images and masks in the pairs list
        pair['img'] = img
        pair['mask'] = mask
    
    # Test out the pixel spacing and image sizes    
    # print(pixel_spaces)
    # print(image_sizes)
    return pairs

def convert_to_PIL_multi(pairs, color_map, img_size=(320, 320)):
    """
    Convert a list of image-mask pairs to PIL format for multi-class segmentation.

    Args:
        pairs (list): A list of dictionaries containing image and mask paths.
        color_map (dict): A color mapping for labels.
        img_size (tuple, optional): The size to which images and masks should be resized. Default is (320, 320).

    Returns:
        list: A list of dictionaries with PIL images.
    """
    image_sizes = []
    # Loop through the dataset
    for pair in pairs:
        # Grab the image and mask path for each pair
        img_path = pair["img"]
        mask = pair["mask"]
        
        # Read the dicom file for image
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
        
        # Read the mask file for mask
        if mask is not None:
            mask_canva = np.zeros((img_size[0], img_size[1]))
            
            # Stack the mask it consist more than 1 binary mask
            for path in mask:
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, img_size)
                m_map = m > 128
                
                path_split = path.split("/")
                region = path_split[-2]
                label = color_map[region]
                
                mask_canva[m_map] = label
            
            mask = Image.fromarray(mask_canva.astype(np.uint8))
            mask = mask.resize(img_size)
            mask = mask.convert("L")
            # The outcome of the mask would be an array with label 0-7
            # Not an image
            
        # Update the images and masks in the pairs list
        pair['img'] = img
        pair['mask'] = mask
    
    return pairs

def display_multi_mask(mask, id2label, color_map):
    """
    Display a multi-label mask using colors from a color map.

    Args:
        mask (numpy.ndarray): The multi-label mask.
        id2label (list): A list of label names.
        color_map (dict): A color mapping for labels.

    Returns:
        PIL.Image: The color-coded mask image.
    """
    mask = np.asarray(mask)
    mask_canva = np.zeros((mask.shape[0], mask.shape[1], 3))  # Empty array to fill in with colour
    # Substitude the label with the color
    for i in range(len(id2label)):
        label = id2label[i]
        color = color_map[label]    # pick the colour base on the label
        mask_canva[mask == i] = color   # fill in the colour
    return Image.fromarray(mask_canva.astype(np.uint8)) # convert to PIL image
        
# ===================== Dataset & DataLoader ===================== #
# Custom Pytorch Dataset Class
class PatientDataset2D(torch.utils.data.Dataset):
    def __init__(self, pairs, transform=None):
        """
        Initialize a custom pytorch patient dataset.

        Args:
            pairs (list): A list of dictionaries containing image and mask paths.
            transform (callable, optional): A transform to apply to the images and masks. Default is None.
        """
        self.pairs = pairs
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: The transformed image.
            torch.Tensor: The transformed mask.
        """
        img_mask_pair = self.pairs[index]
        img = img_mask_pair["img"]
        mask = img_mask_pair["mask"]
        
        # Data Augmentation and Normalization as Tensors
        if self.transform:
            img, mask = random_rotation(img, mask)  # Rotation
            
            img = colour_jitter(img)    # Colour Jitter
            img = self.transform(img)   # To Tensor
            img = normalization(img)    # Normalization
            
            if mask is not None:
                mask = self.transform(mask) # To Tensor
            return img, mask
        else:
            return img, mask
        
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.pairs)


def create_dataloader(patient, transform, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=None):
    """
    Create a DataLoader for a patient dataset.

    Args:
        patient (list): A list of patient data.
        transform (callable): A transform to apply to the images and masks.
        batch_size (int, optional): The batch size. Default is 4.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is False.
        num_workers (int, optional): Number of workers for data loading. Default is 0.
        pin_memory (bool, optional): Whether to use pinned memory. Default is False.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Default is False.
        collate_fn (callable, optional): Custom collate function. Default is None.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the patient dataset.
    """
    # Create a dataset from the patient data
    dataset = PatientDataset2D(patient, transform)
    # Create a pytorch dataloader from custom dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate_fn)
    return dataloader


# ===================== Collate Functions ===================== #
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