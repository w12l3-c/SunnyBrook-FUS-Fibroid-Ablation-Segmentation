# ===================================================================================================
# File description:
# ------------------
# Streamlit app for demoing the segmentation model
# ===================================================================================================


# ======================= Imports =======================
import streamlit as st
import torch
import torchvision

import sys
import os
import re
import cv2
import pydicom

import numpy as np
from PIL import Image
import time

from SpineSeg.model import create_Unet, load_model_torch, accuracy_dice

# ======================= Streamlit =======================
st.title("Regional MRI Segmentation for MRgFUS")

dicom_folder = st.file_uploader("Upload a DICOM Folder", type=["dir", "dcm"], accept_multiple_files=True)
dicom_folder.sort(key=lambda x: x.name)

region = st.selectbox("Select a Model", ["None", "Spine", "Skin", 'Muscle', 'Bowel', 'HipL', 'HipR'])
device = st.selectbox("Select a Device", ["CPU", "GPU"])
if region != "None":
    model, transform = create_Unet(3, 2)
    model = load_model_torch(model, f"trained_models/{region}.pth")
    if device == "GPU":
        device = 'cuda'
        model = model.to(device)
    else:
        device = 'cpu'
    model.eval()
    st.write("--------------------------------------------------")
    
mask_button = st.checkbox("Mask Avaliable?")
mask_folder = None
if mask_button:
    try:
        mask_folder = st.file_uploader("Upload a Mask Folder", type=["dir", "png"], accept_multiple_files=True)
        mask_folder.sort(key=lambda x: x.name)
    except Exception as e:
        st.error("Error reading mask folder: " + str(e))
    st.write("--------------------------------------------------")
    
st.write("--------------------------------------------------")
    
if dicom_folder:
    slice = st.slider("DICOM File Slice", 0, len(dicom_folder)-1, 50)

    try:
        patient_slice = dicom_folder[slice]
        patient_slice = pydicom.dcmread(patient_slice)
        
        img = patient_slice.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.uint8)
        
        # Inference
        if region != "None":
            with torch.inference_mode():
                img = Image.fromarray(img)
                resized_img = img.resize((320, 320))
                transformed_img = transform(resized_img)
                transformed_img = transformed_img.unsqueeze(0)
                
                if device == "GPU":
                    transformed_img = transformed_img.to(device)
                    
                start_time = time.time()
                pred = model(transformed_img)
                inference_time = time.time() - start_time
                
                st.text(f"Inference Time: {inference_time:.5f}s")
                
                pred = torch.softmax(pred, dim=1).argmax(dim=1).float()
                
                if mask_folder is not None:
                    for mask in mask_folder:
                        if int(re.findall(r'\d+', mask.name)[0]) == slice:
                            mask = Image.open(mask)
                            mask = mask.convert('L').resize(320, 320)
                            
                            mask_edge = cv2.Canny(np.array(mask), 100, 200)
                            mask_edge_red = np.zeros((mask_edge.shape[0], mask_edge.shape[1], 3))
                            mask_edge_red[mask_edge > 0] = [255, 0, 0]
                            mask_edge_red = mask_edge_red.astype(np.uint8)
                            
                            mask_red = np.zeros((mask.size[1], mask.size[0], 3))
                            mask_map = np.asarray(mask) > 128
                            mask_red[mask_map] = [255, 0, 0]
                            mask_red = mask_red.astype(np.uint8)
                            
                            break
                    
                    totensor = torchvision.transforms.ToTensor()
                    acc_dice = accuracy_dice(pred, totensor(mask).to(device))
                
                pred = pred.squeeze().cpu().numpy() * 255
                pred = cv2.resize(pred, (320, 320))
                
                pred_edge = cv2.Canny(pred.astype(np.uint8), 100, 200)
                
                pred_edge_green = np.zeros((pred_edge.shape[0], pred_edge.shape[1], 3))
                pred_edge_green[pred_edge > 0] = [0, 255, 0]
                pred_edge_green = pred_edge_green.astype(np.uint8)
                
                pred_green = np.zeros((320, 320, 3))
                pred_map = pred > 0.5
                pred_green[pred_map] = [0, 255, 0]
                pred_green = pred_green.astype(np.uint8)
                
                if mask_folder is not None:
                    edge_overlay = mask_edge_red + pred_edge_green
                    edge_overlay = edge_overlay.astype(np.uint8)
            
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption=f"{patient_slice.PatientName} - {slice}", use_column_width=True)
            if region != "None":
                st.image(pred_green, caption=f"Prediction", use_column_width=True)
                if mask_folder is not None:
                    st.image(mask_red, caption=f"Ground Truth", use_column_width=True)
                    st.image(edge_overlay, caption=f"Edge Overlay", use_column_width=True)
                    st.text(f"Dice Accuracy: {acc_dice:.5f}")
                
                save_path = st.text_input("Save Path", value=f"{patient_slice.PatientName}_{slice}_pred")
                with open(f"{save_path}.png", "wb") as f:
                    f.write(pred_green)
                
                with open(f"{save_path}.png", "rb") as f:
                    st.download_button(
                        label="Download prediction",
                        data=f,
                        file_name=f"{save_path}.png"
                    )
                    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        st.error(f"Error line{exc_tb.tb_lineno}: {str(e)}")
        
    st.write("--------------------------------------------------")
        
