import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from patient import Patient
from vnet import *

def confidence(tensor):
    avg = 0
    for i in range(tensor.shape[0]):
        high = torch.max(tensor[i]) 
        avg += high
    return avg / tensor.shape[0]

def display_volume(logits, confidence, input_size=(128, 128, 128)):
    # Convert logits back to a volume 
    volume = (torch.softmax(logits, dim=1).argmax(dim=1)).view(input_size).cpu().numpy()
    
    # Plot 3D volume
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a mask for voxels with values above the threshold
    mask = volume > 0.5
    ax.voxels(mask, facecolors='red', edgecolor='k')    # Voxels
    
    # Set the labels
    ax.set_xlabel('L')
    ax.set_ylabel('P')
    ax.set_zlabel('S')
    ax.set_title(f'Confidence: {confidence:.2f}%')
    
    plt.show()
    plt.cla()

def inference(model, device, img_stack, input_size=(128, 128, 128)):
    img_stack = torch.unsqueeze(torch.unsqueeze(img_stack, 0), 0)
    img_stack = F.interpolate(img_stack, size=input_size, mode='trilinear', align_corners=False)
    img_stack = img_stack.to(device)
    
    model.eval()
    with torch.inference_mode():
        y_logits = model(img_stack)
        conf = confidence(y_logits)
        conf = conf*100
        
        display_volume(y_logits, conf, input_size)
        
        save = input("Save volume? (y/n): ")
        if save.lower == 'y':
            volume = ((torch.softmax(y_logits, dim=1).argmax(dim=1)).view(input_size)).numpy()
            torch.save(volume, 'volume.pt')
        
        
        
        
        
        
        