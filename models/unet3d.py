"""
Lightweight 3D U-Net Model
Architecture with grouped/depthwise separable convolutions, residual connections
Starting channels: 16 → 32 → 64 → 128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GroupedConv3d(nn.Module):
    """Grouped 3D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=8, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, groups=groups, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with instance normalization and LeakyReLU"""
    def __init__(self, in_channels, out_channels, use_depthwise_separable=True,
                 use_grouped=True, groups=8, dropout_p=0.1):
        super().__init__()
        
        # First convolution
        if use_depthwise_separable:
            self.conv1 = DepthwiseSeparableConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        elif use_grouped and groups > 1 and in_channels >= groups and out_channels >= groups:
            self.conv1 = GroupedConv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        
        # Second convolution
        if use_depthwise_separable:
            self.conv2 = DepthwiseSeparableConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        elif use_grouped and groups > 1 and out_channels >= groups:
            self.conv2 = GroupedConv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        else:
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else None
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.relu2(out)
        
        return out


class DownBlock(nn.Module):
    """Downsampling block with max pooling"""
    def __init__(self, in_channels, out_channels, use_depthwise_separable=True,
                 use_grouped=True, groups=8, dropout_p=0.1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels,
                                       use_depthwise_separable=use_depthwise_separable,
                                       use_grouped=use_grouped,
                                       groups=groups,
                                       dropout_p=dropout_p)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.res_block(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, in_channels, out_channels, use_depthwise_separable=True,
                 use_grouped=True, groups=8, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels,
                                       use_depthwise_separable=use_depthwise_separable,
                                       use_grouped=use_grouped,
                                       groups=groups,
                                       dropout_p=dropout_p)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            # Pad or crop to match skip connection
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


class Lightweight3DUNet(nn.Module):
    """
    Lightweight 3D U-Net for lesion segmentation
    
    Architecture:
        - Encoder: 16 → 32 → 64 → 128 channels
        - Decoder: 128 → 64 → 32 → 16 channels
        - Features: Grouped/depthwise separable convs, residual connections, InstanceNorm, LeakyReLU
    """
    def __init__(self, in_channels=1, out_channels=1, start_channels=16,
                 encoder_channels=[16, 32, 64, 128],
                 use_depthwise_separable=True, use_grouped=True, groups=8,
                 dropout_p=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        
        # Initial convolution
        self.init_conv = ResidualBlock(in_channels, encoder_channels[0],
                                       use_depthwise_separable=use_depthwise_separable,
                                       use_grouped=False,  # First layer uses regular conv
                                       groups=groups,
                                       dropout_p=dropout_p)
        
        # Encoder
        self.down1 = DownBlock(encoder_channels[0], encoder_channels[1],
                               use_depthwise_separable=use_depthwise_separable,
                               use_grouped=use_grouped, groups=groups, dropout_p=dropout_p)
        self.down2 = DownBlock(encoder_channels[1], encoder_channels[2],
                               use_depthwise_separable=use_depthwise_separable,
                               use_grouped=use_grouped, groups=groups, dropout_p=dropout_p)
        self.down3 = DownBlock(encoder_channels[2], encoder_channels[3],
                               use_depthwise_separable=use_depthwise_separable,
                               use_grouped=use_grouped, groups=groups, dropout_p=dropout_p)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(encoder_channels[3], encoder_channels[3],
                                        use_depthwise_separable=use_depthwise_separable,
                                        use_grouped=use_grouped, groups=groups,
                                        dropout_p=dropout_p)
        
        # Decoder
        self.up1 = UpBlock(encoder_channels[3], encoder_channels[2],
                          use_depthwise_separable=use_depthwise_separable,
                          use_grouped=use_grouped, groups=groups, dropout_p=dropout_p)
        self.up2 = UpBlock(encoder_channels[2], encoder_channels[1],
                          use_depthwise_separable=use_depthwise_separable,
                          use_grouped=use_grouped, groups=groups, dropout_p=dropout_p)
        self.up3 = UpBlock(encoder_channels[1], encoder_channels[0],
                          use_depthwise_separable=use_depthwise_separable,
                          use_grouped=use_grouped, groups=groups, dropout_p=dropout_p)
        
        # Output convolution
        self.out_conv = nn.Conv3d(encoder_channels[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output
        x = self.out_conv(x)
        x = self.sigmoid(x)
        
        return x
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def test_model():
    """Test model instantiation and forward pass"""
    model = Lightweight3DUNet(
        in_channels=1,
        out_channels=1,
        start_channels=16,
        encoder_channels=[16, 32, 64, 128],
        use_depthwise_separable=True,
        use_grouped=True,
        groups=8,
        dropout_p=0.1
    )
    
    # Test forward pass
    x = torch.randn(2, 1, 48, 48, 48)  # batch_size=2, patch_size=48x48x48
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {model.count_parameters()}")
    
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    print("Model test passed!")


if __name__ == "__main__":
    test_model()
