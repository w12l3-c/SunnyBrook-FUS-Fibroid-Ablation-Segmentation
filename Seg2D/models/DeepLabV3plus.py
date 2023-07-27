import torch
from torchinfo import summary
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_depth=5,                # depth of encoder
    encoder_weights="imagenet",       # `imagenet` pretrained weights
    in_channels=1,
    classes=1,
)
# When using a Loss_fn that expect logits like CrossEntropy, activation must be none.

print(summary(model, input_size=(1, 1, 320, 320), verbose=0))
