import gradio as gr
import numpy as np
import cv2
import torch
import torchvision
import huggingface_hub

from model import create_Unet
from timeit import default_timer as timer

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

unet, unet_transform = create_Unet(in_channels=3, num_classes=2)
unet.load_state_dict(
  torch.load(
    f="/mnt/HDD_1TB/Wallace/Code/Seg2D/trained_models/Unet_Spine_2023-08-01_18:02:56.pth",
    map_location=torch.device(device),  
  )
)

example_path = '/mnt/HDD_1TB/Wallace/Code/Demo/SpineSeg/example'
example_list = [os.path.join(example_path, example) for example in os.listdir(example_path)]
title = "Region Segmentation for FUS Fibroid Ablation"
description = "Consist of a model for each region(skin, fat, muscle, spine, bowel and hips) in order to create a 3D model of the patient for calculating transducer bloackage during the Focused Ultrasound treatment."
article = "Created by Wallace Lee during Sunnybrook's Internship 2023"

device = 'cpu'

def predict(img):
    start_time = timer()
    resized_img = img.resize((320, 320))
    transformed_img = unet_transform(resized_img).unsqueeze(0)
    transformed_img = transformed_img.to(device)
    
    unet.to(device)
    unet.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(unet(transformed_img), dim=1).argmax(dim=1).float()
        pred = pred_probs.squeeze().cpu().numpy() * 255
        pred = cv2.resize(pred, (img.size[0], img.size[1]))

    overlay = cv2.addWeighted(np.asarray(img), 0.5, pred, 0.5, 0)
    pred_time = round(timer() - start_time, 5)

    return img, pred, overlay, pred_time


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=[gr.Image(type="pil", label="Original Image"), 
             gr.Image(label="Predicted Mask"), 
             gr.Image(label="Overlay"),
             gr.Number(label="Prediction time (s)")
        ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

demo.launch(share=True)





