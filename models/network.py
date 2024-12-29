import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            padding=dilation, groups=in_channels, 
            stride=stride, bias=False, dilation=dilation
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CustomNet(nn.Module):
    def __init__(self, num_classes=10, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # C1 block - Initial convolution with moderate channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),  # RF: 3
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),  # RF: 5->10
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # C2 block with large Dilated Convolution for bigger RF
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=8, dilation=8, bias=False),  # RF: 10->42
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # RF: 42->50
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # C3 block with Depthwise Separable Conv
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 96, stride=1),  # RF: 50->52
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),  # RF: 52->60
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        
        # C4 block with squeeze-excitation
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 40, kernel_size=1, bias=False),  # RF: 60 (1x1 doesn't change RF)
            nn.BatchNorm2d(40),
            nn.ReLU(),
            SEBlock(40),
        )
        
        # Global Average Pooling and FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(40, num_classes)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 