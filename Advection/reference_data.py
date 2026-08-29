import os

import numpy as np


def RBF_np(x1, x2, params):
    output_scale, lengthscales = params
    diffs = (x1[:, None] / lengthscales) - (x2[None, :] / lengthscales)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


f_np = lambda x: np.sin(np.pi * x)
g_np = lambda t: np.sin(np.pi * t / 2.0)


def eval_velocity(gp_sample, x, xmin=0.0, xmax=1.0):
    x = np.asarray(x)
    X = np.linspace(xmin, xmax, gp_sample.shape[0])
    velocity = np.interp(x.reshape(-1), X, gp_sample)
    velocity = velocity - velocity.min() + 1.0
    return velocity.reshape(x.shape)


def _velocity_nodes(gp_sample, xmin=0.0, xmax=1.0):
    x_nodes = np.linspace(xmin, xmax, gp_sample.shape[0], dtype=np.float64)
    velocity = np.asarray(gp_sample, dtype=np.float64).reshape(-1)
    velocity = velocity - velocity.min() + 1.0
    return x_nodes, velocity


def _travel_time_build(gp_sample, xmin=0.0, xmax=1.0):
    x_nodes, vel_nodes = _velocity_nodes(gp_sample, xmin=xmin, xmax=xmax)
    dx = np.diff(x_nodes)
    a0 = vel_nodes[:-1]
    a1 = vel_nodes[1:]
    slope = (a1 - a0) / dx

    tau_nodes = np.zeros_like(x_nodes)
    local = np.empty_like(dx)
    mask = np.abs(slope) < 1e-14
    local[mask] = dx[mask] / a0[mask]
    local[~mask] = np.log(a1[~mask] / a0[~mask]) / slope[~mask]
    tau_nodes[1:] = np.cumsum(local)

    return {
        "x_nodes": x_nodes,
        "vel_nodes": vel_nodes,
        "dx": dx,
        "slope": slope,
        "tau_nodes": tau_nodes,
    }


def _tau_of_x(cache, x):
    x_nodes = cache["x_nodes"]
    vel_nodes = cache["vel_nodes"]
    dx = cache["dx"]
    slope = cache["slope"]
    tau_nodes = cache["tau_nodes"]

    x = np.asarray(x, dtype=np.float64)
    x_flat = np.clip(x.reshape(-1), x_nodes[0], x_nodes[-1])
    idx = np.searchsorted(x_nodes, x_flat, side="right") - 1
    idx = np.clip(idx, 0, len(x_nodes) - 2)

    delta = x_flat - x_nodes[idx]
    a0 = vel_nodes[idx]
    s = slope[idx]
    out = tau_nodes[idx].copy()
    mask = np.abs(s) < 1e-14
    out[mask] += delta[mask] / a0[mask]
    out[~mask] += np.log((a0[~mask] + s[~mask] * delta[~mask]) / a0[~mask]) / s[~mask]
    return out.reshape(x.shape)


def _x_of_tau(cache, tau):
    x_nodes = cache["x_nodes"]
    vel_nodes = cache["vel_nodes"]
    slope = cache["slope"]
    tau_nodes = cache["tau_nodes"]

    tau = np.asarray(tau, dtype=np.float64)
    tau_flat = np.clip(tau.reshape(-1), tau_nodes[0], tau_nodes[-1])
    idx = np.searchsorted(tau_nodes, tau_flat, side="right") - 1
    idx = np.clip(idx, 0, len(tau_nodes) - 2)

    dtau = tau_flat - tau_nodes[idx]
    a0 = vel_nodes[idx]
    s = slope[idx]
    out = x_nodes[idx].copy()
    mask = np.abs(s) < 1e-14
    out[mask] += a0[mask] * dtau[mask]
    out[~mask] += a0[~mask] * (np.exp(s[~mask] * dtau[~mask]) - 1.0) / s[~mask]
    return out.reshape(tau.shape)


def sample_gp_sample(seed, length_scale=0.2, N=512, xmin=0.0, xmax=1.0):
    rng = np.random.default_rng(seed)
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF_np(X, X, (1.0, length_scale))
    L = np.linalg.cholesky(K + 1e-10 * np.eye(N))
    gp_sample = L @ rng.standard_normal(N)
    return gp_sample.astype(np.float64)


def build_case_from_gp_sample(gp_sample, Nx=100, Nt=100, m=100):
    x = np.linspace(0.0, 1.0, Nx, dtype=np.float64)
    t = np.linspace(0.0, 1.0, Nt, dtype=np.float64)
    x_sensor = np.linspace(0.0, 1.0, m, dtype=np.float64)
    cache = _travel_time_build(gp_sample)
    tau_x = _tau_of_x(cache, x)

    TT = t[None, :]
    Tau = tau_x[:, None]
    mask_init = TT <= Tau
    x0 = _x_of_tau(cache, np.maximum(Tau - TT, 0.0))
    tb = TT - Tau
    UU = np.where(mask_init, f_np(x0), g_np(tb))
    u_sensor = eval_velocity(gp_sample, x_sensor).reshape(-1)

    XX, TT = np.meshgrid(x, t, indexing="ij")
    X_ref = np.hstack([XX.reshape(-1, 1), TT.reshape(-1, 1)])
    S_ref = UU.reshape(-1, 1)
    ux_ref = np.repeat(u_sensor[:, None], Nt, axis=1).reshape(-1, 1)

    return {
        "gp_sample": gp_sample.astype(np.float64),
        "x": x.astype(np.float64),
        "t": t.astype(np.float64),
        "solution": UU.astype(np.float64),
        "u_sensor": u_sensor.astype(np.float64),
        "X_ref": X_ref.astype(np.float64),
        "S_ref": S_ref.astype(np.float64),
        "ux_ref": ux_ref.astype(np.float64),
        "Nx": Nx,
        "Nt": Nt,
        "m": m,
    }


def _case_cache_path(seed, length_scale, Nx, Nt, m):
    cache_dir = os.path.join(os.path.dirname(__file__), "CaseCache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"case_seed{int(seed):04d}_ls{float(length_scale):.4f}_nx{int(Nx)}_nt{int(Nt)}_m{int(m)}.npz",
    )


def generate_case(seed, length_scale=0.2, Nx=100, Nt=100, m=100):
    cache_path = _case_cache_path(
        seed=seed, length_scale=length_scale, Nx=Nx, Nt=Nt, m=m
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return {
            "gp_sample": data["gp_sample"].astype(np.float64),
            "x": data["x"].astype(np.float64),
            "t": data["t"].astype(np.float64),
            "solution": data["solution"].astype(np.float64),
            "u_sensor": data["u_sensor"].astype(np.float64),
            "X_ref": data["X_ref"].astype(np.float64),
            "S_ref": data["S_ref"].astype(np.float64),
            "ux_ref": data["ux_ref"].astype(np.float64),
            "Nx": int(data["Nx"]),
            "Nt": int(data["Nt"]),
            "m": int(data["m"]),
            "seed": int(data["seed"]),
            "length_scale": float(data["length_scale"]),
        }

    gp_sample = sample_gp_sample(seed=seed, length_scale=length_scale)
    case = build_case_from_gp_sample(gp_sample, Nx=Nx, Nt=Nt, m=m)
    case["seed"] = int(seed)
    case["length_scale"] = float(length_scale)
    np.savez_compressed(
        cache_path,
        gp_sample=case["gp_sample"],
        x=case["x"],
        t=case["t"],
        solution=case["solution"],
        u_sensor=case["u_sensor"],
        X_ref=case["X_ref"],
        S_ref=case["S_ref"],
        ux_ref=case["ux_ref"],
        Nx=np.int64(case["Nx"]),
        Nt=np.int64(case["Nt"]),
        m=np.int64(case["m"]),
        seed=np.int64(case["seed"]),
        length_scale=np.float64(case["length_scale"]),
    )
    return case


def sample_training_data(case, P_train=1000, Q_train=20000, seed=0):
    rng = np.random.default_rng(seed)

    x_bc1 = np.zeros((P_train // 2, 1))
    x_bc2 = rng.random((P_train // 2, 1))
    t_bc1 = rng.random((P_train // 2, 1))
    t_bc2 = np.zeros((P_train // 2, 1))

    X_bc = np.vstack([np.hstack([x_bc1, t_bc1]), np.hstack([x_bc2, t_bc2])])
    s_bc = np.vstack([g_np(t_bc1), f_np(x_bc2)])

    x_r = rng.uniform(0.0, 1.0, size=(Q_train, 1))
    t_r = rng.uniform(0.0, 1.0, size=(Q_train, 1))
    ux_r = eval_velocity(case["gp_sample"], x_r)
    X_r = np.hstack([x_r, t_r])

    return {
        "X_bc": X_bc.astype(np.float32),
        "s_bc": s_bc.astype(np.float32),
        "X_r": X_r.astype(np.float32),
        "ux_r": ux_r.astype(np.float32),
    }
