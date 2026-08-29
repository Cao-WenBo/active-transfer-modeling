import os

import numpy as np


def RBF_np(x1, x2, params):
    output_scale, lengthscales = params
    diffs = (x1[:, None] / lengthscales) - (x2[None, :] / lengthscales)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def sample_gp_sample(seed, length_scale=0.2, N=512):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.0, 1.0, N)[:, None]
    K = RBF_np(X, X, (1.0, length_scale))
    L = np.linalg.cholesky(K + 1e-10 * np.eye(N))
    gp_sample = L @ rng.standard_normal(N)
    return gp_sample.astype(np.float64)


def eval_forcing(gp_sample, x):
    X = np.linspace(0.0, 1.0, gp_sample.shape[0])
    x = np.asarray(x)
    forcing = np.interp(x.reshape(-1), X, gp_sample)
    return forcing.reshape(x.shape)


def _solve_tridiagonal(lower, diag, upper, rhs):
    n = diag.shape[0]
    c_prime = np.empty(n - 1, dtype=diag.dtype)
    d_prime = np.empty(n, dtype=rhs.dtype)

    denom = diag[0]
    c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom

    x = np.empty(n, dtype=rhs.dtype)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def solve_ADR_np(gp_sample, Nx, Nt):
    xmin, xmax = 0.0, 1.0
    tmin, tmax = 0.0, 1.0

    k_fn = lambda x: 0.01 * np.ones_like(x, dtype=np.float64)
    g_fn = lambda u: 0.01 * u**2
    dg_fn = lambda u: 0.02 * u

    x = np.linspace(xmin, xmax, Nx, dtype=np.float64)
    t = np.linspace(tmin, tmax, Nt, dtype=np.float64)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h * h

    k = k_fn(x)
    f = eval_forcing(gp_sample, x)

    n_interior = Nx - 2
    base_diag = np.full(n_interior, 8.0 * h2 / dt + 8.0 * k[1], dtype=np.float64)
    base_off = np.full(n_interior - 1, -4.0 * k[1], dtype=np.float64)
    c_diag = np.full(n_interior, 8.0 * h2 / dt - 8.0 * k[1], dtype=np.float64)
    c_off = np.full(n_interior - 1, 4.0 * k[1], dtype=np.float64)

    u = np.zeros((Nx, Nt), dtype=np.float64)

    for i in range(Nt - 1):
        gi = g_fn(u[1:-1, i])
        dgi = dg_fn(u[1:-1, i])
        h2dgi = 4.0 * h2 * dgi
        A_diag = base_diag - h2dgi
        b1 = 8 * h2 * (f[1:-1] + gi)
        rhs = b1 + (c_diag - h2dgi) * u[1:-1, i]
        rhs[:-1] += c_off * u[2:-1, i]
        rhs[1:] += c_off * u[1:-2, i]
        u[1:-1, i + 1] = _solve_tridiagonal(base_off, A_diag, base_off, rhs)

    return x, t, u


def _case_cache_path(seed, length_scale, Nx, Nt, m, refine):
    cache_dir = os.path.join(os.path.dirname(__file__), "CaseCache")
    os.makedirs(cache_dir, exist_ok=True)
    file_name = (
        f"seed{seed:05d}_ls{length_scale:.4f}_Nx{Nx}_Nt{Nt}_m{m}_ref{refine}.npz"
    )
    return os.path.join(cache_dir, file_name)


def build_case_from_gp_sample(gp_sample, Nx=100, Nt=100, m=100, refine=4):
    Nx_hr = refine * (Nx - 1) + 1
    Nt_hr = refine * (Nt - 1) + 1
    x_hr, t_hr, solution_hr = solve_ADR_np(gp_sample, Nx_hr, Nt_hr)

    x = x_hr[::refine]
    t = t_hr[::refine]
    solution = solution_hr[::refine, ::refine]
    u_sensor = eval_forcing(gp_sample, np.linspace(0.0, 1.0, m, dtype=np.float64))

    XX, TT = np.meshgrid(x, t, indexing="ij")
    X_ref = np.hstack([XX.reshape(-1, 1), TT.reshape(-1, 1)])
    S_ref = solution.reshape(-1, 1)
    forcing_ref = np.repeat(eval_forcing(gp_sample, x)[:, None], Nt, axis=1).reshape(
        -1, 1
    )

    return {
        "gp_sample": gp_sample.astype(np.float64),
        "x": x.astype(np.float64),
        "t": t.astype(np.float64),
        "solution": solution.astype(np.float64),
        "u_sensor": u_sensor.astype(np.float64),
        "X_ref": X_ref.astype(np.float64),
        "S_ref": S_ref.astype(np.float64),
        "forcing_ref": forcing_ref.astype(np.float64),
        "Nx": int(Nx),
        "Nt": int(Nt),
        "m": int(m),
    }


def generate_case(seed, length_scale=0.2, Nx=100, Nt=100, m=100, refine=4):
    cache_path = _case_cache_path(seed, length_scale, Nx, Nt, m, refine)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        return {
            "gp_sample": cached["gp_sample"],
            "x": cached["x"],
            "t": cached["t"],
            "solution": cached["solution"],
            "u_sensor": cached["u_sensor"],
            "X_ref": cached["X_ref"],
            "S_ref": cached["S_ref"],
            "forcing_ref": cached["forcing_ref"],
            "Nx": int(cached["Nx"][()]),
            "Nt": int(cached["Nt"][()]),
            "m": int(cached["m"][()]),
            "seed": int(cached["seed"][()]),
            "length_scale": float(cached["length_scale"][()]),
        }

    gp_sample = sample_gp_sample(seed=seed, length_scale=length_scale)
    case = build_case_from_gp_sample(gp_sample, Nx=Nx, Nt=Nt, m=m, refine=refine)
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
        forcing_ref=case["forcing_ref"],
        Nx=np.array(case["Nx"], dtype=np.int64),
        Nt=np.array(case["Nt"], dtype=np.int64),
        m=np.array(case["m"], dtype=np.int64),
        seed=np.array(case["seed"], dtype=np.int64),
        length_scale=np.array(case["length_scale"], dtype=np.float64),
    )
    return case


def sample_training_data(case, P_train=3000, Q_train=20000, seed=0):
    rng = np.random.default_rng(seed)

    n_side = P_train // 3
    n_init = P_train - 2 * n_side

    x_bc1 = np.zeros((n_side, 1))
    x_bc2 = np.ones((n_side, 1))
    x_bc3 = rng.random((n_init, 1))
    t_bc1 = rng.random((2 * n_side, 1))
    t_bc2 = np.zeros((n_init, 1))

    X_bc = np.vstack(
        [
            np.hstack([x_bc1, t_bc1[:n_side]]),
            np.hstack([x_bc2, t_bc1[n_side:]]),
            np.hstack([x_bc3, t_bc2]),
        ]
    )
    s_bc = np.zeros((P_train, 1), dtype=np.float32)

    x_r = rng.random((Q_train, 1))
    t_r = rng.random((Q_train, 1))
    X_r = np.hstack([x_r, t_r]).astype(np.float32)
    forcing_r = eval_forcing(case["gp_sample"], x_r).astype(np.float32)

    return {
        "X_bc": X_bc.astype(np.float32),
        "s_bc": s_bc,
        "X_r": X_r,
        "forcing_r": forcing_r,
    }
