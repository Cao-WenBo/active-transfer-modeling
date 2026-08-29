import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers, reference_points):
        super().__init__()
        self.register_buffer("X_mean", reference_points.mean(0, keepdim=True))
        self.register_buffer(
            "X_std", reference_points.std(0, keepdim=True).clamp_min(1e-6)
        )
        self.layers = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )

    def forward(self, x):
        h = (x - self.X_mean) / self.X_std
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i + 1 < len(self.layers):
                h = torch.tanh(h)
        return h


def vector_to_param_dict(model, vector):
    out, pointer = {}, 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        out[name] = vector[pointer : pointer + count].view_as(parameter)
        pointer += count
    return out
