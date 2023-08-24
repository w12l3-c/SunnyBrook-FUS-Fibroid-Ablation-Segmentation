# ===================================================================================================
# File description:
# ------------------
# Streamlit app for demoing the segmentation model
# ===================================================================================================


# ======================= Imports =======================
import streamlit as st
import torch
import torchvision

import os
import cv2
import pydicom
import dicom2nifti
import numpy as np
from PIL import Image
import time

from SpineSeg.model import create_Unet, load_model_torch

# ======================= Streamlit =======================
dicom_folder = st.file_uploader("Upload a DICOM Folder", type=["dir", "dcm"], accept_multiple_files=True)
dicom_folder.sort(key=lambda x: x.name)

region = st.selectbox("Select a Model", ["None", "Spine", "Skin", 'Muscle', 'Bowel', 'HipL', 'HipR'])
device = st.selectbox("Select a Device", ["CPU", "GPU"])
if region != "None":
    model, transform = create_Unet(3, 2)
    model = load_model_torch(model, f"trained_models/{region}.pth")
    if device == "GPU":
        model = model.to(device)
    model.eval()
    st.write("Model Loaded")
    st.write("--------------------------------------------------")
    
st.write("--------------------------------------------------")
    
if dicom_folder:
    slice = st.slider("DICOM File Slice", 0, len(dicom_folder)-1, 50)
    
    
    try:
        patient_slice = dicom_folder[slice]
        patient_slice = pydicom.dcmread(patient_slice)
        
        st.write("--------------------------------------------------")
        
        img = patient_slice.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        
        st.image(img, caption=f"{patient_slice.PatientName} - {slice}", use_column_width=True)
        
        # Inference
        if region != "None":
            with torch.inference_mode():
                resized_img = cv2.resize(img, (320, 320))
                transformed_img = transform(resized_img)
                transformed_img = transformed_img.unsqueeze(0)
                if device == "GPU":
                    transformed_img = transformed_img.to(device)
                    
                start_time = time.time()
                pred = model(transformed_img)
                inference_time = time.time() - start_time
                
                pred = torch.softmax(pred, dim=1).argmax(dim=1).float()
                
                
    

        
    except Exception as e:
        st.error("Error reading DICOM file: " + str(e))
        
mask_button = st.checkbox("Mask Avaliable?")
if mask_button:
    try:
        mask_folder = st.file_uploader("Upload a Mask Folder", type=["dir", "png"], accept_multiple_files=True)
        mask_folder.sort(key=lambda x: x.name)
    except Exception as e:
        st.error("Error reading mask folder: " + str(e))
    st.write("--------------------------------------------------")