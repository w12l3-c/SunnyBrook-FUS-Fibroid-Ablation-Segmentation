import numpy as np
import cv2
import os
import pydicom
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageOps

from sklearn.model_selection import train_test_split

from save_load import *
from dataloader import *
from models import KMean, Unet
from model_inference_fn import unet_inference
from spine import test_spine_dataset
from hipl import test_hipl_dataset

sagittal_model_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Spine_2023-08-01_18:02:56.pth"
coronal_model_path = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_HipL_2023-08-10_16.pth"

sag_model = Unet.auto_UNET(in_channels=3, num_classes=2)
sag_model = load_model_torch(sag_model, sagittal_model_path)

for i in range(2):
    data = test_spine_dataset[i+np.random.randint(0, len(test_spine_dataset))]
    image = data['img']
    image = image.convert('RGB')
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    sag_model.eval()
    with torch.inference_mode():
        img = image.convert('RGB')
        resized_img = img.resize((320, 320))
        transformed_img = transform(resized_img)
        start_time = time.time()
        logits = sag_model(transformed_img.unsqueeze(0))
        ml_time = time.time() - start_time
        pred = torch.softmax(logits, dim=1).argmax(dim=1)
        pred = pred.squeeze(0).numpy()
        pred = cv2.resize(pred, (img.size[0], img.size[1]))
        pred_edge = cv2.Canny(pred.astype(np.uint8), 100, 200)
        pred_edge_mask = pred_edge > 0
        
        pred_mask = pred > 0
        pred_green = np.zeros((pred.shape[0], pred.shape[1], 3))
        pred_green[pred_mask] = [0, 255, 0]
    
    np_image = np.asarray(image)
    for i in range(1):
        jitter = cv2.convertScaleAbs(np_image, alpha=np.random.randint(1, 4), beta=np.random.randint(10, 20))
        jitter = cv2.cvtColor(jitter, cv2.COLOR_RGB2GRAY)
        inverse = cv2.bitwise_not(jitter)
        np_image = np.concatenate((np_image, jitter.reshape(np_image.shape[0], np_image.shape[1], 1), inverse.reshape(np_image.shape[0], np_image.shape[1], 1)), axis=2)
    start_time = time.time()
    kmean_pred = KMean.kmeans_segmentation(np.asarray(np_image), num_class=8)
    kmean_time = time.time() - start_time
    
    ax[0].imshow(image)
    ax[1].imshow(pred_green, cmap='gray')
    ax[2].imshow(kmean_pred, cmap='gray')
    ax[3].imshow(kmean_pred, alpha=0.8, cmap='gray')
    ax[3].imshow(pred_green, alpha=0.5)
    fig.suptitle(f"Kmean Unet time diff: {(kmean_time - ml_time):.4f}s")
    plt.show()
        
cor_model = Unet.auto_UNET(in_channels=3, num_classes=2)
cor_model = load_model_torch(cor_model, coronal_model_path)

for i in range(2):
    data = test_hipl_dataset[i+np.random.randint(0, len(test_hipl_dataset))]
    image = data['img']
    image = image.convert('RGB')
    image = image.resize((160, 160))
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    sag_model.eval()
    with torch.inference_mode():
        transformed_img = transform(image)
        logits = cor_model(transformed_img.unsqueeze(0))
        pred = torch.softmax(logits, dim=1).argmax(dim=1)
        pred = pred.squeeze(0).numpy()
        pred = cv2.resize(pred, (image.size[0], image.size[1]))
        pred_edge = cv2.Canny(pred.astype(np.uint8), 100, 200)
        pred_edge_mask = pred_edge > 0
        
        pred_mask = pred > 0
        pred_green = np.zeros((pred.shape[0], pred.shape[1], 3))
        pred_green[pred_mask] = [0, 255, 0]
        
    kmean_pred = KMean.kmeans_segmentation(np.asarray(image), num_class=3)
    kmean_pred[kmean_pred>10] = 255
    
    ax[0].imshow(image)
    ax[1].imshow(pred_green, cmap='gray')
    ax[2].imshow(kmean_pred, cmap='gray')
    ax[3].imshow(kmean_pred, alpha=0.8, cmap='gray')
    ax[3].imshow(pred_green, alpha=0.5)
    plt.show()
    
    

