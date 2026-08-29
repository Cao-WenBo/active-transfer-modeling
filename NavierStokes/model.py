import torch
import torch.nn as nn

import config


class Net(nn.Module):
    def __init__(self, layers, reference_points):
        super().__init__()
        self.register_buffer("T_scale", reference_points[:, 2:3].max().clamp_min(1e-6))
        self.fourier_kernel = nn.Parameter(
            config.FOURIER_EMBED_SCALE * torch.randn(5, config.FOURIER_EMBED_DIM // 2)
        )
        self.layers = nn.ModuleList(
            [
                nn.Linear(layers[index], layers[index + 1])
                for index in range(len(layers) - 1)
            ]
        )

    def forward(self, points):
        periodic = torch.cat(
            [
                torch.cos(points[:, 0:1]),
                torch.sin(points[:, 0:1]),
                torch.cos(points[:, 1:2]),
                torch.sin(points[:, 1:2]),
                points[:, 2:3] / self.T_scale,
            ],
            dim=1,
        )
        projected = periodic @ self.fourier_kernel
        state = torch.cat([torch.cos(projected), torch.sin(projected)], dim=1)
        for index, layer in enumerate(self.layers):
            state = layer(state)
            if index + 1 < len(self.layers):
                state = torch.tanh(state)
        return state
