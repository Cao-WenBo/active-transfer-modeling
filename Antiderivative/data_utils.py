from functools import lru_cache
import numpy as np
import torch

import config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rbf(x1, x2, length_scale):
    diff = x1[:, None] / length_scale - x2[None, :] / length_scale
    return np.exp(-0.5 * np.sum(diff * diff, axis=2))


def gp_sample(seed, n=512):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)[:, None]
    chol = np.linalg.cholesky(rbf(x, x, config.LENGTH_SCALE) + 1e-10 * np.eye(n))
    return (chol @ rng.standard_normal(n)).astype(np.float64)


def evaluate_input(values, x):
    nodes = np.linspace(0.0, 1.0, len(values))
    x = np.asarray(x)
    return np.interp(x.reshape(-1), nodes, values).reshape(x.shape)


def integrate_piecewise_linear(values, x):
    nodes = np.linspace(0.0, 1.0, len(values))
    dx = np.diff(nodes)
    cumulative = np.zeros_like(nodes)
    cumulative[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * dx)
    flat = np.clip(np.asarray(x).reshape(-1), 0.0, 1.0)
    idx = np.clip(np.searchsorted(nodes, flat, side="right") - 1, 0, len(nodes) - 2)
    delta = flat - nodes[idx]
    slope = (values[idx + 1] - values[idx]) / (nodes[idx + 1] - nodes[idx])
    return (cumulative[idx] + values[idx] * delta + 0.5 * slope * delta**2).reshape(
        np.asarray(x).shape
    )


@lru_cache(maxsize=256)
def generate_case(seed):
    path = config.CASE_CACHE / f"case_seed{int(seed):05d}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    values = gp_sample(int(seed))
    x = np.linspace(0.0, 1.0, config.N_REF)
    sensors = np.linspace(0.0, 1.0, config.M)
    case = {
        "seed": np.int64(seed),
        "gp_sample": values,
        "x": x,
        "x_sensor": sensors,
        "u_sensor": evaluate_input(values, sensors),
        "rhs_ref": evaluate_input(values, x),
        "X_ref": x[:, None],
        "S_ref": integrate_piecewise_linear(values, x)[:, None],
    }
    config.CASE_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **case)
    return case


def sample_training_data(case):
    x_bc = np.zeros((config.P_TRAIN, 1), np.float32)
    x_r = np.linspace(0.0, 1.0, config.Q_TRAIN, dtype=np.float32)[:, None]
    return {
        "X_bc": x_bc,
        "s_bc": np.zeros_like(x_bc),
        "X_r": x_r,
        "rhs_r": evaluate_input(case["gp_sample"], x_r).astype(np.float32),
    }
