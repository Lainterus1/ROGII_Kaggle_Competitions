"""Temporal Convolutional Network with causal convolutions + BatchNorm."""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class CausalConv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.chomp = Chomp1d(self.padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.chomp(out)
        return self.relu(out)


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1dBlock(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1dBlock(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn(out + residual)
        return self.relu(out)


class TCNModel(nn.Module):
    def __init__(self, input_size: int, num_channels: list[int] | None = None,
                 kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256]
        layers: list[nn.Module] = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size,
                                   dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn(x)
        return self.head(out[:, :, -1])
