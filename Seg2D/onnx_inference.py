# ======================================================================== #
# File Description:
# ------------------
# This is for when you want even faster inference time 
# Which is converting the tensorflow or pytorch model in onnx model
#
# It doesn't show that much improve in this case as it usually 
# see significant difference in LLM and Large Diffusion model
#
# I have NOT Debug this code yet
# ======================================================================== #

# =================== Imports =================== #
import cv2
import numpy as np
import torch
import torch.onnx as onnx
import onnxruntime as ort

from models import Unet


def convert_and_optimize_model(pytorch_model, input_shape, onnx_model_path):
    # Set the model to evaluation mode
    pytorch_model.eval()

    # Dummy input for tracing (adjust the shape and type according to your model's input)
    dummy_input = torch.randn(input_shape)

    # Export the PyTorch model to ONNX format
    torch.onnx.export(pytorch_model, dummy_input, onnx_model_path)

    # Optional: Optimize the ONNX model using ONNX Runtime's optimization tools (if needed)
    optimized_model = onnx.load(onnx_model_path)
    onnx.save(optimized_model, onnx_model_path)

def preprocess_input(image_path, preprocess):
    # Load and preprocess the input image
    image = cv2.imread(image_path)
    preprocessed_image = preprocess(image)
    return preprocessed_image

def postprocess_output(output):
    # Perform any postprocessing on the output to get the final segmentation mask
    postprocessed_output = torch.softmax(output, dim=1).argmax(dim=1).float()
    return postprocessed_output

def segment_image(image_path, shape=(8, 3, 320, 320)):
    # Load model and weights
    model = Unet.auto_UNET(3, 2)
    model.load_state_dict(torch.load("path_to_model_weights.pth"))
    
    # Convert and optimize the model to ONNX format
    onnx_model_path = "path_to_optimized_model.onnx"
    convert_and_optimize_model(model, shape, onnx_model_path)
    
    # Load ONNX model for the segmentation model
    session = ort.InferenceSession(onnx_model_path)

    input_data = preprocess_input(image_path)
    input_name = session.get_inputs()[0].name
    inputs = {input_name: np.expand_dims(input_data, 0)}

    outputs = session.run(None, inputs)
    segmentation_mask = postprocess_output(outputs)

    return segmentation_mask

if __name__ == "__main__":
    image_paths = []

    for image_path in image_paths:
        segmentation_mask = segment_image(image_path)
        # Do something with the segmentation mask (e.g., save, visualize, etc.)
