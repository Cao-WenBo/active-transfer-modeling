import torch
from model import vector_to_param_dict


def state_explicit(model, theta, X):
    """Return only the Darcy-required fields [u, u_x, u_y, u_xx, u_yy]."""
    params = vector_to_param_dict(model, theta)
    n, dim = X.shape
    h = (X - model.X_mean) / model.X_std
    eye = torch.eye(dim, device=X.device, dtype=X.dtype)
    dh = eye[None].expand(n, -1, -1) / model.X_std.unsqueeze(-1)
    ddh = torch.zeros(n, dim, dim, dim, device=X.device, dtype=X.dtype)
    for i in range(len(model.layers)):
        W, b = params[f"layers.{i}.weight"], params[f"layers.{i}.bias"]
        z = h @ W.T + b
        dz = torch.einsum("oi,nij->noj", W, dh)
        ddz = torch.einsum("oi,nijk->nojk", W, ddh)
        if i + 1 < len(model.layers):
            h = torch.tanh(z)
            fp = 1 - h.square()
            fpp = -2 * h * fp
            ddh = fp[..., None, None] * ddz + fpp[..., None, None] * (
                dz[..., :, None] * dz[..., None, :]
            )
            dh = fp[..., None] * dz
        else:
            h, dh, ddh = z, dz, ddz
    raw, draw, ddraw = h[:, 0], dh[:, 0], ddh[:, 0]
    x, y = X[:, 0], X[:, 1]
    ax, ay = x * (1 - x), y * (1 - y)
    d = ax * ay
    dx, dy = (1 - 2 * x) * ay, ax * (1 - 2 * y)
    dxx, dyy = -2 * ay, -2 * ax
    u = d * raw
    ux = dx * raw + d * draw[:, 0]
    uy = dy * raw + d * draw[:, 1]
    uxx = dxx * raw + 2 * dx * draw[:, 0] + d * ddraw[:, 0, 0]
    uyy = dyy * raw + 2 * dy * draw[:, 1] + d * ddraw[:, 1, 1]
    return torch.stack((u, ux, uy, uxx, uyy), dim=1)
