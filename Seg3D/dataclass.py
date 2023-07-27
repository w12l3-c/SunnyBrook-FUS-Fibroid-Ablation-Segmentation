import torch
import torch.nn as nn
 
import torchvision 
 
import numpy as np
import os

from patient import Patient
from utils import stack3d, flatten_mask, transform

# ---------------------- Dataset ---------------------- #
# Input a paitent list with each element as a Patient object or dict
class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, patient_list, transform=None):
        self.patient_list = patient_list
        self.transform = transform
        
    def __getitem__(self, index):
        self.patient = self.patient_list[index]
        self.img = self.patient['img']
        self.volume = self.patient['mask']
        
        if self.img.dim() == 3:
            if self.transform:
                # Torchvision.transform
                self.img = self.transform(self.img)
                self.volume = self.transform(self.volume)
        elif self.img.dim() == 4:
            if self.transform:
                # Interpolation
                self.img, self.volume = transform(self.img, self.volume, 128, 128, 128)
        return self.img, self.volume
        
    def __len__(self):
        return len(self.patient_list)
    

# ---------------------- Data Loader ---------------------- #
def create_dataloader(img_list, mask_list, batch_size, shuffle, num_workers, pin_memory):
    dataset = PatientDataset(img_list, mask_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)    
    return dataloader