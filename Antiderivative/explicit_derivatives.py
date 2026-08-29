import torch
from model import vector_to_param_dict


def state_explicit(model, params_vector, x):
    """One batched pass returning exactly [s, s_x]."""
    params = vector_to_param_dict(model, params_vector)
    h = (x - model.X_mean) / model.X_std
    hx = torch.ones_like(h) / model.X_std[0, 0]
    for index in range(len(model.layers)):
        weight = params[f"layers.{index}.weight"]
        bias = params[f"layers.{index}.bias"]
        z = h @ weight.T + bias
        zx = hx @ weight.T
        if index + 1 < len(model.layers):
            h = torch.tanh(z)
            hx = (1.0 - h.square()) * zx
        else:
            h, hx = z, zx
    return torch.cat([h, hx], dim=1)
