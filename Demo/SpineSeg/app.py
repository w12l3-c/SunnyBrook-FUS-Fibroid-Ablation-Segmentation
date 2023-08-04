import gradio as gr
import numpy as np
import cv2
import torch
import torchvision
import huggingface_hub
import time
import os
import pydicom
from PIL import Image

from model import create_Unet
from timeit import default_timer as timer

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

unet, unet_transform = create_Unet(in_channels = 3, num_classes = 2)
unet.load_state_dict(
    torch.load(
        f = "/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Spine_2023-08-01_18:02:56.pth",
        map_location = torch.device(device),
    )
)

example_path = '/mnt/HDD_1TB/Wallace/Code/Demo/SpineSeg/example'
example_list = [os.path.join(example_path, example) for example in os.listdir(example_path)]
title = "Region Segmentation for FUS Fibroid Ablation"
description = "Consist of a model for each region(skin, fat, muscle, spine, bowel and hips) in order to create a 3D model of the patient for calculating transducer bloackage during the Focused Ultrasound treatment."
article = "Created by Wallace Lee during Sunnybrook's Internship 2023"


def predict(image):
    unet.to(device)    
    unet.eval()
    with torch.no_grad():
        image = image.convert('RGB')
        resized_image = image.resize((320, 320))
        transformed_image = unet_transform(resized_image).to(device)

        start_time = time.time()
        logits = unet(transformed_image.unsqueeze(0))
        pred = torch.softmax(logits, dim = 1).argmax(dim = 1).float()
        end_time = time.time()
        
        pred = pred.squeeze().cpu().numpy() * 255
        pred = cv2.resize(pred, (image.size[0], image.size[1]))
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        pred = pred.astype(np.uint8)
        image = np.asarray(image).astype(np.uint8)

        inference_time = end_time - start_time
    
        overlay = cv2.addWeighted(image, 0.5, pred, 0.5, 0)

        return image, pred, overlay, inference_time
    
def dicom_input(dicom_file):
    dicom_data = pydicom.dcmread(dicom_file)
    img = dicom_data.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    return img
    


demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type='pil', label='Input Image'),
    outputs = [gr.Image(type = "pil", label = "Original Image"),
        gr.Image(label = "Predicted Mask"),
        gr.Image(label = "Overlay"),
        gr.Number(label = "Prediction time (s)")
    ],
    examples = example_list,
    title = title,
    description = description,
    article = article,
    
)

demo.launch(share = True)