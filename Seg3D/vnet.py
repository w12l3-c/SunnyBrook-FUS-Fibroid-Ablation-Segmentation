import torch 
import torch.nn as nn
import torch.nn.functional as F

# import torchinfo
# from torchinfo import summary

# import torchvision

# import segmentation_models_pytorch as smp

# ------------------- Vnet ------------------- #
# A Larger model means more time to train
# A Larger model also means more memory in file size
# A Larger size of tensor for volume means more memory occupy in GPU

# Placeholder function for dropout layer
def passthrough(x, **kwargs):
    return x

# Activation function
def ELUCons(elu, nchan):
    # Exponential Linear Unit or Parametric Exponential Linear Unit
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, nChan, elu):
        super(ConvBlock, self).__init__()
        self.relu1 = ELUCons(elu, nChan)
        self.conv1 = nn.Conv3d(nChan, nChan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(nChan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

# Making Conv Stacks
def make_conv_stack(nChan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(ConvBlock(nChan, elu))
    return nn.Sequential(*layers)

# Input Layer
class InputLayer(nn.Module):
    def __init__(self, outChans, elu):
        super(InputLayer, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)   # ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels (for summing with 16 channel out)
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), dim=1)    
        out = self.relu1(torch.add(out, x16))   # sumwise addition 
        return out

# Downsample Layer
class DownTransition(nn.Module):
    # nConvs are number of conv layers
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans) # ContBatchNorm3d(outChans)
        self.do1 = passthrough  # placeholder dropout
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()   
        self.ops = make_conv_stack(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))  # downsample
        out = self.do1(down)                            # dropout
        out = self.ops(out)                             # conv layers
        out = self.relu2(torch.add(out, down))          # sumwise addition
        return out


# Upsample Layer
class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = make_conv_stack(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)                               # dropout           
        skipxdo = self.do2(skipx)                       # dropout for skip connection
        out = self.relu1(self.bn1(self.up_conv(out)))   # upsample
        xcat = torch.cat((out, skipxdo), 1)             # add skip connection
        out = self.ops(xcat)                            # conv layers
        out = self.relu2(torch.add(out, xcat))          # sumwise addition
        return out

# Output Layer
class OutputLayer(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputLayer, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax   # log softmax for NLL loss
        else:
            self.softmax = F.softmax       # softmax for cross entropy loss

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(1, 2, 3, 4, 0).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        # softmax
        out = self.softmax(out, dim=1)
        # treat channel 0 as the predicted output
        return out
    
# VNet
class VNet(nn.Module):
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.input_layer = InputLayer(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.output_layer = OutputLayer(32, elu, nll)

    def forward(self, x):
        # Input
        out16 = self.input_layer(x)
        
        # Downsample
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        
        # Upsample with skip connections
        out = self.up_tr256(out256, out128) 
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        
        # Output
        out = self.output_layer(out)
        return out
    
    
# ---------------------- Loss Functions ---------------------- #
# Dice Coefficient
def dice_coefficient(pred, target):
    smooth = 1e-5
    intersection = torch.sum(pred * target)
    union = torch.sum(pred**2) + torch.sum(target**2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Dice Loss Class
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        dice = dice_coefficient(pred, target)
        dice_loss = 1 - dice
        return dice_loss
    
    
# ---------------------- Accuracy Functions ---------------------- #
# Accuracy (Intersection)
def accuracy(pred, target):
    intersection = torch.sum(pred * target)
    return intersection
    
    
# ---------------------- Test ---------------------- #
if __name__ == "__main__":
    # Test if a tensor can be passed through the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = VNet().to(device)
    # Best shape to fit in the GPU is (1, 1, )
    # Need to modify the dicom images to fit this shape
    example_tensor = torch.rand(1, 1, 144, 144, 144).to(device)
    result = model(example_tensor)
    print(result)
    print(result.shape)
    
    # Test for model's dimension by printing out the shape of each layer
    example_tensor = torch.rand(1, 144, 320, 320)
    input_tensor = nn.Conv3d(1, 16, kernel_size=5, padding=2)(example_tensor)
    print('Input:')
    print(input_tensor.shape)

    down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)(input_tensor)
    out = nn.Conv3d(32, 32, kernel_size=5, padding=2)(down1)
    out = torch.add(out, down1)
    print('Down 1:')
    print(out.shape)

    down1 = nn.Conv3d(32, 64, kernel_size=2, stride=2)(out)
    out = nn.Conv3d(64, 64, kernel_size=5, padding=2)(down1)
    out = torch.add(out, down1)
    print('Down 2:')
    print(out.shape)

    down1 = nn.Conv3d(64, 128, kernel_size=2, stride=2)(out)
    out = nn.Conv3d(128, 128, kernel_size=5, padding=2)(down1)
    out = torch.add(out, down1)
    print('Down 3:')
    print(out.shape)

    down1 = nn.Conv3d(128, 256, kernel_size=2, stride=2)(out)
    out = nn.Conv3d(256, 256, kernel_size=5, padding=2)(down1)
    out = torch.add(out, down1)
    print('Down 4:')
    print(out.shape)
    
    up = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)(out)
    cat = torch.cat((up, torch.rand(128, 18, 40, 40)))
    print('Up 1:')
    print(up.shape)
    print(cat.shape)

    up = nn.ConvTranspose3d(256, 64, kernel_size=2, stride=2)(cat)
    cat = torch.cat((up, torch.rand(64, 36, 80, 80)))
    print('Up 2:')
    print(up.shape)
    print(cat.shape)

    up = nn.ConvTranspose3d(128, 32, kernel_size=2, stride=2)(cat)
    cat = torch.cat((up, torch.rand(32, 72, 160, 160)))
    print('Up 3:')
    print(up.shape)
    print(cat.shape)

    up = nn.ConvTranspose3d(64, 16, kernel_size=2, stride=2)(cat)
    cat = torch.cat((up, torch.rand(16, 144, 320, 320)))
    print('Up 4:')
    print(up.shape)
    print(cat.shape)
        
    print('Output:')
    output = nn.Conv3d(32, 2, kernel_size=5, padding=2)(cat)
    print(output.shape)

    output2 = nn.Conv3d(2, 2, kernel_size=1)(output)
    print(output2.shape)

    flatten = output2.permute(1, 2, 3, 4, 0).contiguous()
    flatten = flatten.view(flatten.numel() // 2, 2)
    print(flatten.shape)

    soft = torch.nn.functional.softmax(flatten, dim=1)
    print(soft.shape)
    arg = torch.argmax(soft, dim=1)
    
    # Summary of the model
    # This somehow doens't work in vscode 
    # Because torchinfo doesn't work
    # Run this in Colab or Jupyter Notebook instead
    # print(summary(model, 
    #               input_size=(1, 1, 144, 320, 320), 
    #               verbose=0, 
    #               col_names=["input_size", "output_size", "num_params", "trainable"]
    #               )
    #       )
    

    