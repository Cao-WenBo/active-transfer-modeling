import torch
from model import vector_to_param_dict


def state_explicit(model, theta, x):
    """One batched pass returning only [s,s_t,s_xx]."""
    p = vector_to_param_dict(model, theta)
    h = (x - model.X_mean) / model.X_std
    hx = torch.zeros_like(h)
    ht = torch.zeros_like(h)
    hxx = torch.zeros_like(h)
    hx[:, 0] = 1 / model.X_std[0, 0]
    ht[:, 1] = 1 / model.X_std[0, 1]
    for i in range(len(model.layers)):
        w = p[f"layers.{i}.weight"]
        b = p[f"layers.{i}.bias"]
        z = h @ w.T + b
        zx = hx @ w.T
        zt = ht @ w.T
        zxx = hxx @ w.T
        if i + 1 < len(model.layers):
            h = torch.tanh(z)
            d1 = 1 - h.square()
            d2 = -2 * h * d1
            hx = d1 * zx
            ht = d1 * zt
            hxx = d2 * zx.square() + d1 * zxx
        else:
            h, hx, ht, hxx = z, zx, zt, zxx
    return torch.cat([h, ht, hxx], 1)
