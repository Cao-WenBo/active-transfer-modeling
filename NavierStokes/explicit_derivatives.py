import torch

from data_utils import FORCE_N


def vector_to_param_dict(model, theta_vec):
    param_dict = {}
    pointer = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param_dict[name] = theta_vec[pointer : pointer + numel].view_as(param)
        pointer += numel
    return param_dict


def _linear_first(A, Ax, Ay, weight, bias=None):
    Z = A @ weight.T
    if bias is not None:
        Z = Z + bias
    return Z, Ax @ weight.T, Ay @ weight.T


def _activation_first(Z, Zx, Zy, kind):
    if kind == "tanh":
        A = torch.tanh(Z)
        phi1 = 1.0 - A * A
    elif kind == "cos":
        A = torch.cos(Z)
        phi1 = -torch.sin(Z)
    elif kind == "sin":
        A = torch.sin(Z)
        phi1 = torch.cos(Z)
    else:
        raise ValueError(kind)
    return A, phi1 * Zx, phi1 * Zy


def _periodic_feature_first(X):
    n_points = X.shape[0]
    A = X.new_zeros(n_points, 5)
    Ax = X.new_zeros(n_points, 5)
    Ay = X.new_zeros(n_points, 5)
    x = X[:, 0]
    y = X[:, 1]
    A[:, 0] = torch.cos(x)
    A[:, 1] = torch.sin(x)
    A[:, 2] = torch.cos(y)
    A[:, 3] = torch.sin(y)
    A[:, 4] = X[:, 2]
    Ax[:, 0] = -torch.sin(x)
    Ax[:, 1] = torch.cos(x)
    Ay[:, 2] = -torch.sin(y)
    Ay[:, 3] = torch.cos(y)
    return A, Ax, Ay


def kolmogorov_net_first_from_params(model, params_vector, X):
    params = vector_to_param_dict(model, params_vector)
    A, Ax, Ay = _periodic_feature_first(X)
    A[:, 4] = A[:, 4] / model.T_scale
    Z, Zx, Zy = _linear_first(A, Ax, Ay, params["fourier_kernel"].T)
    A_cos, Ax_cos, Ay_cos = _activation_first(Z, Zx, Zy, "cos")
    A_sin, Ax_sin, Ay_sin = _activation_first(Z, Zx, Zy, "sin")
    A = torch.cat([A_cos, A_sin], dim=1)
    Ax = torch.cat([Ax_cos, Ax_sin], dim=1)
    Ay = torch.cat([Ay_cos, Ay_sin], dim=1)

    n_layers = len(model.layers)
    for layer_id in range(n_layers):
        weight = params[f"layers.{layer_id}.weight"]
        bias = params[f"layers.{layer_id}.bias"]
        A, Ax, Ay = _linear_first(A, Ax, Ay, weight, bias)
        if layer_id < n_layers - 1:
            A, Ax, Ay = _activation_first(A, Ax, Ay, "tanh")
    return A, Ax, Ay


def _linear_compact(
    A, Ax, Ay, At, Axx, Axy, Ayy, Axt, Ayt, Axxx, Axxy, Axyy, Ayyy, weight, bias=None
):
    Z = A @ weight.T
    if bias is not None:
        Z = Z + bias
    return (
        Z,
        Ax @ weight.T,
        Ay @ weight.T,
        At @ weight.T,
        Axx @ weight.T,
        Axy @ weight.T,
        Ayy @ weight.T,
        Axt @ weight.T,
        Ayt @ weight.T,
        Axxx @ weight.T,
        Axxy @ weight.T,
        Axyy @ weight.T,
        Ayyy @ weight.T,
    )


def _activation_compact(
    Z, Zx, Zy, Zt, Zxx, Zxy, Zyy, Zxt, Zyt, Zxxx, Zxxy, Zxyy, Zyyy, kind
):
    if kind == "tanh":
        A = torch.tanh(Z)
        phi1 = 1.0 - A * A
        phi2 = -2.0 * A * phi1
        phi3 = -2.0 + 8.0 * A * A - 6.0 * A**4
    elif kind == "cos":
        A = torch.cos(Z)
        phi1 = -torch.sin(Z)
        phi2 = -A
        phi3 = torch.sin(Z)
    elif kind == "sin":
        A = torch.sin(Z)
        phi1 = torch.cos(Z)
        phi2 = -A
        phi3 = -phi1
    else:
        raise ValueError(kind)

    Ax = phi1 * Zx
    Ay = phi1 * Zy
    At = phi1 * Zt
    Axx = phi2 * Zx * Zx + phi1 * Zxx
    Axy = phi2 * Zx * Zy + phi1 * Zxy
    Ayy = phi2 * Zy * Zy + phi1 * Zyy
    Axt = phi2 * Zx * Zt + phi1 * Zxt
    Ayt = phi2 * Zy * Zt + phi1 * Zyt
    Axxx = phi3 * Zx * Zx * Zx + 3.0 * phi2 * Zxx * Zx + phi1 * Zxxx
    Axxy = phi3 * Zx * Zx * Zy + phi2 * (Zxx * Zy + 2.0 * Zxy * Zx) + phi1 * Zxxy
    Axyy = phi3 * Zx * Zy * Zy + phi2 * (2.0 * Zxy * Zy + Zyy * Zx) + phi1 * Zxyy
    Ayyy = phi3 * Zy * Zy * Zy + 3.0 * phi2 * Zyy * Zy + phi1 * Zyyy
    return A, Ax, Ay, At, Axx, Axy, Ayy, Axt, Ayt, Axxx, Axxy, Axyy, Ayyy


def _periodic_feature_compact(X, T_scale):
    n_points = X.shape[0]
    A = X.new_zeros(n_points, 5)
    Ax = X.new_zeros(n_points, 5)
    Ay = X.new_zeros(n_points, 5)
    At = X.new_zeros(n_points, 5)
    Axx = X.new_zeros(n_points, 5)
    Axy = X.new_zeros(n_points, 5)
    Ayy = X.new_zeros(n_points, 5)
    Axt = X.new_zeros(n_points, 5)
    Ayt = X.new_zeros(n_points, 5)
    Axxx = X.new_zeros(n_points, 5)
    Axxy = X.new_zeros(n_points, 5)
    Axyy = X.new_zeros(n_points, 5)
    Ayyy = X.new_zeros(n_points, 5)

    x = X[:, 0]
    y = X[:, 1]
    A[:, 0] = torch.cos(x)
    A[:, 1] = torch.sin(x)
    A[:, 2] = torch.cos(y)
    A[:, 3] = torch.sin(y)
    A[:, 4] = X[:, 2] / T_scale
    Ax[:, 0] = -torch.sin(x)
    Ax[:, 1] = torch.cos(x)
    Ay[:, 2] = -torch.sin(y)
    Ay[:, 3] = torch.cos(y)
    At[:, 4] = 1.0 / T_scale
    Axx[:, 0] = -torch.cos(x)
    Axx[:, 1] = -torch.sin(x)
    Ayy[:, 2] = -torch.cos(y)
    Ayy[:, 3] = -torch.sin(y)
    Axxx[:, 0] = torch.sin(x)
    Axxx[:, 1] = -torch.cos(x)
    Ayyy[:, 2] = torch.sin(y)
    Ayyy[:, 3] = -torch.cos(y)
    return A, Ax, Ay, At, Axx, Axy, Ayy, Axt, Ayt, Axxx, Axxy, Axyy, Ayyy


def kolmogorov_net_compact_from_params(model, params_vector, X):
    params = vector_to_param_dict(model, params_vector)
    state = _periodic_feature_compact(X, model.T_scale)
    state = _linear_compact(*state, params["fourier_kernel"].T)
    cos_state = _activation_compact(*state, "cos")
    sin_state = _activation_compact(*state, "sin")
    state = tuple(
        torch.cat([cos_state[i], sin_state[i]], dim=1) for i in range(len(cos_state))
    )

    n_layers = len(model.layers)
    for layer_id in range(n_layers):
        weight = params[f"layers.{layer_id}.weight"]
        bias = params[f"layers.{layer_id}.bias"]
        state = _linear_compact(*state, weight, bias)
        if layer_id < n_layers - 1:
            state = _activation_compact(*state, "tanh")
    return state


def kolmogorov_uvw_explicit(model, params_vector, X):
    U, Ux, Uy = kolmogorov_net_first_from_params(model, params_vector, X)
    W = Ux[:, 1:2] - Uy[:, 0:1]
    return torch.cat([U, W], dim=1)


def kolmogorov_pde_residual_explicit(model, params_vector, X_r, Re):
    (
        U,
        Ux,
        Uy,
        Ut,
        Uxx,
        Uxy,
        Uyy,
        Uxt,
        Uyt,
        Uxxx,
        Uxxy,
        Uxyy,
        Uyyy,
    ) = kolmogorov_net_compact_from_params(model, params_vector, X_r)
    u = U[:, 0:1]
    v = U[:, 1:2]
    w_x = Uxx[:, 1:2] - Uxy[:, 0:1]
    w_y = Uxy[:, 1:2] - Uyy[:, 0:1]
    w_t = Uxt[:, 1:2] - Uyt[:, 0:1]
    w_xx = Uxxx[:, 1:2] - Uxxy[:, 0:1]
    w_yy = Uxyy[:, 1:2] - Uyyy[:, 0:1]
    force = -FORCE_N * torch.cos(FORCE_N * X_r[:, 1:2])
    eq1 = w_t + u * w_x + v * w_y - (w_xx + w_yy) / Re - force
    eq2 = Ux[:, 0:1] + Uy[:, 1:2]
    return torch.cat([eq1, eq2], dim=0)


def kolmogorov_pde_features_explicit(model, params_vector, X_r):
    (
        U,
        Ux,
        Uy,
        Ut,
        Uxx,
        Uxy,
        Uyy,
        Uxt,
        Uyt,
        Uxxx,
        Uxxy,
        Uxyy,
        Uyyy,
    ) = kolmogorov_net_compact_from_params(model, params_vector, X_r)
    u = U[:, 0:1]
    v = U[:, 1:2]
    w_x = Uxx[:, 1:2] - Uxy[:, 0:1]
    w_y = Uxy[:, 1:2] - Uyy[:, 0:1]
    w_t = Uxt[:, 1:2] - Uyt[:, 0:1]
    w_xx = Uxxx[:, 1:2] - Uxxy[:, 0:1]
    w_yy = Uxyy[:, 1:2] - Uyyy[:, 0:1]
    div = Ux[:, 0:1] + Uy[:, 1:2]
    return torch.cat([u, v, w_x, w_y, w_t, w_xx, w_yy, div], dim=1)


def kolmogorov_system_explicit(model, params_vector, X_ic, U_ic, X_r, Re):
    r_ic = (kolmogorov_uvw_explicit(model, params_vector, X_ic) - U_ic).reshape(-1, 1)
    r_pde = kolmogorov_pde_residual_explicit(model, params_vector, X_r, Re).reshape(
        -1, 1
    )
    N_ic = r_ic.shape[0]
    N_pde = r_pde.shape[0]
    N_tot = N_ic + N_pde
    w_ic = (10 * N_tot / N_ic) ** 0.5
    w_pde = (1 * N_tot / N_pde) ** 0.5
    return torch.cat([w_ic * r_ic, w_pde * r_pde], dim=0)


def _linear_spatial(state, weight, bias=None):
    output = tuple(value @ weight.T for value in state)
    if bias is not None:
        output = (output[0] + bias, *output[1:])
    return output


def _activation_spatial(state, kind):
    z, zx, zy, zxx, zxy, zyy, zxxx, zxxy, zxyy, zyyy = state
    if kind == "tanh":
        value = torch.tanh(z)
        first = 1.0 - value.square()
        second = -2.0 * value * first
        third = -2.0 + 8.0 * value.square() - 6.0 * value**4
    elif kind == "cos":
        value = torch.cos(z)
        first, second, third = -torch.sin(z), -value, torch.sin(z)
    elif kind == "sin":
        value = torch.sin(z)
        first, second, third = torch.cos(z), -value, -torch.cos(z)
    else:
        raise ValueError(kind)
    return (
        value,
        first * zx,
        first * zy,
        second * zx.square() + first * zxx,
        second * zx * zy + first * zxy,
        second * zy.square() + first * zyy,
        third * zx**3 + 3.0 * second * zxx * zx + first * zxxx,
        third * zx.square() * zy + second * (zxx * zy + 2.0 * zxy * zx) + first * zxxy,
        third * zx * zy.square() + second * (2.0 * zxy * zy + zyy * zx) + first * zxyy,
        third * zy**3 + 3.0 * second * zyy * zy + first * zyyy,
    )


def kolmogorov_state_minimal_explicit(model, params_vector, points):
    """Return only [u,v,w,w_x,w_y,w_xx,w_yy,div] for time marching."""
    params = vector_to_param_dict(model, params_vector)
    n = points.shape[0]
    state = tuple(points.new_zeros(n, 5) for _ in range(10))
    value, dx, dy, dxx, dxy, dyy, dxxx, dxxy, dxyy, dyyy = state
    x, y = points[:, 0], points[:, 1]
    value[:, 0], value[:, 1] = torch.cos(x), torch.sin(x)
    value[:, 2], value[:, 3] = torch.cos(y), torch.sin(y)
    value[:, 4] = points[:, 2] / model.T_scale
    dx[:, 0], dx[:, 1] = -torch.sin(x), torch.cos(x)
    dy[:, 2], dy[:, 3] = -torch.sin(y), torch.cos(y)
    dxx[:, 0], dxx[:, 1] = -torch.cos(x), -torch.sin(x)
    dyy[:, 2], dyy[:, 3] = -torch.cos(y), -torch.sin(y)
    dxxx[:, 0], dxxx[:, 1] = torch.sin(x), -torch.cos(x)
    dyyy[:, 2], dyyy[:, 3] = torch.sin(y), -torch.cos(y)

    state = _linear_spatial(state, params["fourier_kernel"].T)
    cosine = _activation_spatial(state, "cos")
    sine = _activation_spatial(state, "sin")
    state = tuple(
        torch.cat([cosine[index], sine[index]], dim=1) for index in range(len(state))
    )
    for layer_id in range(len(model.layers)):
        state = _linear_spatial(
            state,
            params[f"layers.{layer_id}.weight"],
            params[f"layers.{layer_id}.bias"],
        )
        if layer_id + 1 < len(model.layers):
            state = _activation_spatial(state, "tanh")

    value, dx, dy, dxx, dxy, dyy, dxxx, dxxy, dxyy, dyyy = state
    w = dx[:, 1:2] - dy[:, 0:1]
    wx = dxx[:, 1:2] - dxy[:, 0:1]
    wy = dxy[:, 1:2] - dyy[:, 0:1]
    wxx = dxxx[:, 1:2] - dxxy[:, 0:1]
    wyy = dxyy[:, 1:2] - dyyy[:, 0:1]
    div = dx[:, 0:1] + dy[:, 1:2]
    return torch.cat([value[:, 0:2], w, wx, wy, wxx, wyy, div], dim=1)
