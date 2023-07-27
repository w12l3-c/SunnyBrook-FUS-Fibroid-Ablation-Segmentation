import gradio as gr
import numpy as np
import torch
import torchvision
import huggingface_hub

examples = []
title = ""
description = ""
article = ""

def predict():
    pass


demo = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(),
    outputs=gr.outputs.Image(),
    examples=examples,
    title=title,
    description=description,
    article=article
)

demo.launch(share=True)