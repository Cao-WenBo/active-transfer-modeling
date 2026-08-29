import torch

from model import vector_to_param_dict


def burgers_state_explicit(model, params_vector, x):
    """Return only [u, u_x, u_t, u_xx], the quantities used by Burgers residual."""
    params = vector_to_param_dict(model, params_vector)
    mean, std = model.X_mean, model.X_std
    h = (x - mean) / std
    hx = torch.zeros_like(h)
    ht = torch.zeros_like(h)
    hxx = torch.zeros_like(h)
    hx[:, 0] = 1.0 / std[0, 0]
    ht[:, 1] = 1.0 / std[0, 1]

    for index, _ in enumerate(model.layers):
        weight = params[f"layers.{index}.weight"]
        bias = params[f"layers.{index}.bias"]
        z = h @ weight.T + bias
        zx = hx @ weight.T
        zt = ht @ weight.T
        zxx = hxx @ weight.T
        if index + 1 < len(model.layers):
            h = torch.tanh(z)
            first = 1.0 - h.square()
            second = -2.0 * h * first
            hx = first * zx
            ht = first * zt
            hxx = second * zx.square() + first * zxx
        else:
            h, hx, ht, hxx = z, zx, zt, zxx
    return torch.cat([h, hx, ht, hxx], dim=1)


def burgers_u_explicit(model, params_vector, x):
    return burgers_state_explicit(model, params_vector, x)[:, :1]
