import pytest
import torch
from models.network import CustomNet  # Updated import to match train.py
from torchsummary import summary
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = CustomNet(num_classes=10)  # Updated to match your model
    total_params = count_parameters(model)
    assert total_params < 200000, f"Model has {total_params} parameters, should be less than 200k"

def calculate_rf(model):
    rf = 1
    dilation = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kernel_size = m.kernel_size[0]
            curr_dilation = m.dilation[0]
            rf = rf + ((kernel_size - 1) * curr_dilation * dilation)
            dilation = dilation * m.stride[0]
    return rf

def test_receptive_field():
    model = CustomNet(num_classes=10)
    total_rf = calculate_rf(model)
    assert total_rf > 44, f"Receptive field is {total_rf}, should be more than 44"

def test_architecture_components():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomNet(num_classes=10, device=device).to(device)
    
    # Check for GAP
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model must include Global Average Pooling"
    
    # Check for Depthwise Separable Conv
    has_depthwise = False
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.groups > 1:
            has_depthwise = True
            break
    assert has_depthwise, "Model must include Depthwise Separable Convolution"
    
    # Check for Dilated Conv
    has_dilated = False
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.dilation[0] > 1:
            has_dilated = True
            break
    assert has_dilated, "Model must include Dilated Convolution"

def test_strided_convolutions():
    model = CustomNet(num_classes=10)
    stride2_count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.stride[0] == 2:
            stride2_count += 1
    assert stride2_count >= 3, "Model must have at least 3 convolutions with stride 2" 