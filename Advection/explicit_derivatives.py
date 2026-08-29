import torch
from model import vector_to_param_dict


def state_explicit(model, theta, x):
    p = vector_to_param_dict(model, theta)
    h = (x - model.X_mean) / model.X_std
    hx = torch.zeros_like(h)
    ht = torch.zeros_like(h)
    hx[:, 0] = 1 / model.X_std[0, 0]
    ht[:, 1] = 1 / model.X_std[0, 1]
    for i in range(len(model.layers)):
        w = p[f"layers.{i}.weight"]
        b = p[f"layers.{i}.bias"]
        z = h @ w.T + b
        zx = hx @ w.T
        zt = ht @ w.T
        if i + 1 < len(model.layers):
            h = torch.tanh(z)
            d = 1 - h.square()
            hx = d * zx
            ht = d * zt
        else:
            h, hx, ht = z, zx, zt
    return torch.cat([h, hx, ht], 1)
