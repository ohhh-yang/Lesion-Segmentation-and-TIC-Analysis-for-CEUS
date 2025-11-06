# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""

    def __init__(self, n, weight=False):
        """Initializes a module to sum outputs of layers with number of inputs `n` and optional weighting, supporting 2+
        inputs.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights."""
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Initializes MixConv2d with mixed depth-wise convolutional layers, taking input and output channels (c1, c2),
        kernel sizes (k), stride (s), and channel distribution strategy (equal_ch).
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Performs forward pass by applying SiLU activation on batch-normalized concatenated convolutional layer
        outputs.
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()

        # 通道注意力模块
        self.channel_attention = ChannelAttention(in_channels, reduction)

        # 空间注意力模块
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 通道注意力
        x = self.channel_attention(x)

        # 空间注意力
        x = self.spatial_attention(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        # 通道注意力的实现，使用全局池化（平均池化和最大池化）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的全连接层
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))

        # 融合两种池化结果
        out = avg_out + max_out
        return torch.sigmoid(out) * x  # 注意力加权


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        # 空间注意力的实现，使用卷积生成空间注意力图
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接两种池化结果
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_out)

        return torch.sigmoid(x_out) * x  # 注意力加权


class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_Block, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Squeeze and excitation layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1, bias=False
        )  # Fully connected layer 1
        self.fc2 = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1, bias=False
        )  # Fully connected layer 2
        self.sigmoid = nn.Sigmoid()  # Sigmoid to scale channel weights

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x)  # Global average pooling
        y = self.fc1(y)  # First fully connected layer
        y = F.relu(y)  # ReLU activation
        y = self.fc2(y)  # Second fully connected layer
        y = self.sigmoid(y)  # Sigmoid to scale channels

        return x * y  # Scale input by the attention weights


class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Horizontal and vertical attention
        self.fc_h = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1, bias=False
        )
        self.fc_w = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1, bias=False
        )
        self.fc_out_h = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1, bias=False
        )
        self.fc_out_w = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Horizontal Attention
        h = x.mean(dim=2, keepdim=True)  # Global Average Pooling across height
        h = self.fc_h(h)
        h = F.relu(h)
        h = self.fc_out_h(h)

        # Vertical Attention
        w = x.mean(dim=3, keepdim=True)  # Global Average Pooling across width
        w = self.fc_w(w)
        w = F.relu(w)
        w = self.fc_out_w(w)

        # Apply the attention mechanism
        attn_h = self.sigmoid(h)
        attn_w = self.sigmoid(w)

        return x * attn_h * attn_w  # Apply the horizontal and vertical attentions


class ASFF(nn.Module):
    def __init__(self, level, channels=256, rfb=False, norm_layer=nn.BatchNorm2d):
        """
        Adaptive Spatial Feature Fusion (ASFF) module for feature fusion.
        Args:
            level: Feature level to be fused (P3=8, P4=16, P5=32).
            channels: Number of output channels.
            rfb: Whether to use Receptive Field Block (optional).
            norm_layer: Normalization layer (default is BatchNorm2d).
        """
        super(ASFF, self).__init__()
        self.level = level
        self.inter_dim = channels
        self.dim = [channels, channels, channels]  # Assume P3, P4, P5 input sizes

        # Create scale-specific layers
        if level == 0:  # P3
            self.stride_level_1 = nn.Conv2d(
                self.dim[1], self.inter_dim, 3, stride=2, padding=1
            )
            self.stride_level_2 = nn.Conv2d(
                self.dim[2], self.inter_dim, 3, stride=2, padding=1
            )
        elif level == 1:  # P4
            self.compress_level_0 = nn.Conv2d(self.dim[0], self.inter_dim, 1, stride=1)
            self.stride_level_2 = nn.Conv2d(
                self.dim[2], self.inter_dim, 3, stride=2, padding=1
            )
        elif level == 2:  # P5
            self.compress_level_0 = nn.Conv2d(self.dim[0], self.inter_dim, 1, stride=1)
            self.compress_level_1 = nn.Conv2d(self.dim[1], self.inter_dim, 1, stride=1)

        # Attention weights
        self.weight_level_0 = nn.Conv2d(self.inter_dim, 1, 1, stride=1)
        self.weight_level_1 = nn.Conv2d(self.inter_dim, 1, 1, stride=1)
        self.weight_level_2 = nn.Conv2d(self.inter_dim, 1, 1, stride=1)

        # Final combination layer
        self.combine = nn.Conv2d(self.inter_dim, self.inter_dim, 3, padding=1, stride=1)
        self.act = nn.SiLU()

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            x_level_1_down = self.stride_level_1(x_level_1)
            x_level_2_down = self.stride_level_2(x_level_2)
            x_level_0 = x_level_0
        elif self.level == 1:
            x_level_0_compressed = self.compress_level_0(x_level_0)
            x_level_2_down = self.stride_level_2(x_level_2)
            x_level_1 = x_level_1
        elif self.level == 2:
            x_level_0_compressed = self.compress_level_0(x_level_0)
            x_level_1_compressed = self.compress_level_1(x_level_1)
            x_level_2 = x_level_2

        # Compute attention weights
        weight_0 = self.weight_level_0(x_level_0)
        weight_1 = self.weight_level_1(x_level_1)
        weight_2 = self.weight_level_2(x_level_2)

        print(weight_0.shape, weight_1.shape, weight_2.shape)
        weights = torch.cat([weight_0, weight_1, weight_2], dim=1)
        weights = F.softmax(weights, dim=1)

        # Fuse features
        fused = (
            x_level_0 * weights[:, 0:1, :, :]
            + x_level_1 * weights[:, 1:2, :, :]
            + x_level_2 * weights[:, 2:3, :, :]
        )

        return self.act(self.combine(fused))


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=rates[0],
            dilation=rates[0],
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=rates[1],
            dilation=rates[1],
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=rates[2],
            dilation=rates[2],
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.final_conv = nn.Conv2d(
            out_channels * 5, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn_final = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))

        x5 = self.global_avg_pool(x)
        x5 = self.conv5(x5)
        # H W -> 1 1 此时BatchNorm2D会报错，一个维度至少需要2个数据
        if not (x5.shape[-1] == 1 and x5.shape[-2] == 1):
            x5 = self.bn5(x5)
        x5 = F.interpolate(
            x5, size=x.shape[2:], mode="bilinear", align_corners=False
        )  # 调整回输入大小

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn_final(self.final_conv(x)))
        return x
