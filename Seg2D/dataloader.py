import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from patient import *

import cv2
import pydicom
import datasets
from PIL import Image


colour_jitter = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
])

normalization = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229 , 0.224, 0.225])
])

def convert_to_PIL(pairs):
    for pair in pairs:
        img = pair["img"]
        mask = pair["mask"]
        
        img = pydicom.dcmread(img).pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        img = img.resize((320, 320))
        img = img.convert("RGB")
        
        mask = Image.open(mask)
        mask = mask.resize((320, 320))
        mask = mask.convert("L")
        
        pair['img'] = img
        pair['mask'] = mask
    
    return pairs

class PatientDataset2D(torch.utils.data.Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform
    
    def __getitem__(self, index):
        img_mask_pair = self.pairs[index]
        img = img_mask_pair["img"]
        mask = img_mask_pair["mask"]
        
        if self.transform:
            img = colour_jitter(img)
            img = self.transform(img)
            img = normalization(img)
            mask = self.transform(mask)
            return img, mask
        else:
            return img, mask
        
    def __len__(self):
        return len(self.pairs)

# Input should be the same, the processsing done in init
# patient == [{img:, mask:}]

def create_dataloader(patient, transform, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
    dataset = PatientDataset2D(patient, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    return dataloader