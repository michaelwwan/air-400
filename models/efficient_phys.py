"""EfficientPhys: lightweight TS-CAN-style architecture."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AttentionMask(nn.Module):
    """Attention mask with NaN protection."""

    def __init__(self) -> None:
        """Initialize mask module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Generate normalized mask while guarding against NaNs."""
        # Handle NaN input
        if torch.isnan(x).any():
            print("Warning: NaN input to Attention_mask")
            x = torch.nan_to_num(x, nan=0.0)
            
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        
        # Prevent division by zero
        eps = 1e-8
        safe_xsum = torch.where(xsum > eps, xsum, torch.ones_like(xsum) * eps)
        
        xshape = tuple(x.size())
        output = x / safe_xsum * xshape[2] * xshape[3] * 0.5
        
        # Handle potential NaN in output
        if torch.isnan(output).any():
            print("Warning: NaN output from Attention_mask")
            output = torch.nan_to_num(output, nan=0.0)
            
        return output


class TSM(nn.Module):
    """Temporal shift module reused across models."""

    def __init__(self, n_segment: int = 10, fold_div: int = 3) -> None:
        """Store number of segments and fold division."""
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: Tensor) -> Tensor:
        """Apply temporal shift to tensor `x`."""
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class EfficientPhys(nn.Module):
    """
    EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement

    From: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023)
    Authors: Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff
    """

    def __init__(
        self,
        in_channels: int = 3,
        nb_filters1: int = 32,
        nb_filters2: int = 64,
        kernel_size: int = 3,
        dropout_rate1: float = 0.25,
        dropout_rate2: float = 0.5,
        pool_size: Tuple[int, int] = (2, 2),
        nb_dense: int = 128,
        frame_depth: int = 20,
        img_size: int = 36,
    ) -> None:
        """Initialize the EfficientPhys architecture."""
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                  bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                  bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = AttentionMask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = AttentionMask()
        
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
            
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
        self.batch_norm = nn.BatchNorm2d(self.in_channels)

    def forward(self, inputs: Tensor, params: Optional[Tensor] = None) -> Tensor:
        """Return respiration prediction for the provided clip batch."""
        inputs = torch.diff(inputs, dim=0)
        inputs = self.batch_norm(inputs)

        network_input = self.TSM_1(inputs)
        d1 = torch.tanh(self.motion_conv1(network_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out
