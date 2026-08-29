import math
import os

import numpy as np
import torch

import config


L = 2.0 * math.pi
RE = config.RE
TIME_INTERVAL = config.TIME_INTERVAL
FORCE_N = 4.0
GRF_ALPHA = config.GRF_ALPHA
GRF_TAU = config.GRF_TAU
NX_REF = config.NX
NT_REF = config.SOURCE_NT
CACHE_DIR = str(config.CASE_CACHE)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GaussianRF2d:
    def __init__(
        self,
        s,
        length=L,
        alpha=GRF_ALPHA,
        tau=GRF_TAU,
        device="cpu",
        dtype=torch.float64,
    ):
        self.s = s
        self.length = length
        self.device = device
        self.dtype = dtype
        k = torch.cat(
            (
                torch.arange(0, s // 2, device=device),
                torch.arange(-s // 2, 0, device=device),
            )
        ).type(dtype)
        kx = k.view(-1, 1).repeat(1, s)
        ky = k.view(1, -1).repeat(s, 1)
        self.sqrt_eig = tau ** (alpha - 1.0) * (
            (2.0 * math.pi / length) ** 2 * (kx**2 + ky**2) + tau**2
        ) ** (-alpha / 2.0)
        self.sqrt_eig[0, 0] = 0.0

    def sample(self, seed):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        xi = torch.randn(
            (self.s, self.s), generator=gen, device=self.device, dtype=self.dtype
        )
        return torch.fft.ifft2(self.s * self.sqrt_eig * torch.fft.fft2(xi)).real


class NavierStokes2d:
    def __init__(self, s, length=L, device="cpu", dtype=torch.float64):
        self.s = s
        self.length = length
        self.device = device
        self.dtype = dtype
        freq = torch.cat(
            (
                torch.arange(0, s // 2, device=device),
                torch.arange(-s // 2, 0, device=device),
            )
        ).type(dtype)
        self.kx = freq.view(-1, 1).repeat(1, s)
        self.ky = freq.view(1, -1).repeat(s, 1)
        self.G = (2.0 * math.pi / length) ** 2 * (self.kx**2 + self.ky**2)
        self.inv_lap = self.G.clone()
        self.inv_lap[0, 0] = 1.0
        self.inv_lap = 1.0 / self.inv_lap
        self.dealias = ((self.kx**2 + self.ky**2) <= (s / 3.0) ** 2).type(dtype)
        self.dealias[0, 0] = 0.0

    def stream_function(self, w_h):
        return self.inv_lap * w_h

    def velocity_field(self, w_h):
        psi_h = self.stream_function(w_h)
        u_h = (2.0 * math.pi / self.length) * 1j * self.ky * psi_h
        v_h = -(2.0 * math.pi / self.length) * 1j * self.kx * psi_h
        return torch.fft.ifft2(u_h).real, torch.fft.ifft2(v_h).real

    def nonlinear_term(self, w_h, f_h):
        w = torch.fft.ifft2(w_h).real
        u, v = self.velocity_field(w_h)
        nonlin = (
            -1j
            * (2.0 * math.pi / self.length)
            * (self.kx * torch.fft.fft2(u * w) + self.ky * torch.fft.fft2(v * w))
        )
        return nonlin + f_h

    def advance(self, w, forcing, T, Re, dt=1e-3):
        w_h = torch.fft.fft2(w)
        f_h = torch.fft.fft2(forcing)
        G = self.G / Re
        time = 0.0
        while time < T - 1e-14:
            current_dt = min(dt, T - time)
            n1 = self.nonlinear_term(w_h, f_h)
            w_tilde = (w_h + current_dt * (n1 - 0.5 * G * w_h)) / (
                1.0 + 0.5 * current_dt * G
            )
            n2 = self.nonlinear_term(w_tilde, f_h)
            w_h = (w_h + current_dt * (0.5 * (n1 + n2) - 0.5 * G * w_h)) / (
                1.0 + 0.5 * current_dt * G
            )
            w_h = w_h * self.dealias
            time += current_dt
        return torch.fft.ifft2(w_h).real


def periodic_bilinear(field, points, length=L):
    field = np.asarray(field)
    s = field.shape[-1]
    x = (np.asarray(points)[:, 0] % length) / length * s
    y = (np.asarray(points)[:, 1] % length) / length * s
    i0 = np.floor(x).astype(np.int64) % s
    j0 = np.floor(y).astype(np.int64) % s
    i1 = (i0 + 1) % s
    j1 = (j0 + 1) % s
    wx = x - np.floor(x)
    wy = y - np.floor(y)
    return (
        (1 - wx) * (1 - wy) * field[i0, j0]
        + wx * (1 - wy) * field[i1, j0]
        + (1 - wx) * wy * field[i0, j1]
        + wx * wy * field[i1, j1]
    ).reshape(-1, 1)


def _case_cache_path(seed, Re=RE, nx=NX_REF, nt=NT_REF, T=TIME_INTERVAL):
    ensure_dir(CACHE_DIR)
    return os.path.join(
        CACHE_DIR,
        f"case_forced_periodic_seed{int(seed):05d}_Re{float(Re):.1f}_nx{nx}_nt{nt}_T{float(T):.2f}.npz",
    )


def generate_case(
    seed=0, Re=RE, nx=NX_REF, nt=NT_REF, T=TIME_INTERVAL, spinup=0.0, device="cpu"
):
    path = _case_cache_path(seed, Re=Re, nx=nx, nt=nt, T=T)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    dtype = torch.float64
    solver = NavierStokes2d(nx, L, device=device, dtype=dtype)
    grf = GaussianRF2d(nx, L, device=device, dtype=dtype)
    grid = torch.linspace(0.0, L, nx + 1, device=device, dtype=dtype)[:-1]
    _, yy = torch.meshgrid(grid, grid, indexing="ij")
    force = -FORCE_N * torch.cos(FORCE_N * yy)
    w = grf.sample(seed)
    if spinup > 0.0:
        w = solver.advance(w, force, spinup, Re)
    times = np.linspace(0.0, T, nt, dtype=np.float64)
    fields = []
    last_t = 0.0
    for t in times:
        if t > last_t:
            w = solver.advance(w, force, float(t - last_t), Re)
        u, v = solver.velocity_field(torch.fft.fft2(w))
        fields.append(torch.stack([u, v, w], dim=-1).detach().cpu().numpy())
        last_t = float(t)
    fields = np.stack(fields, axis=0)
    x = np.linspace(0.0, L, nx, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, L, nx, endpoint=False, dtype=np.float64)
    tt, xxn, yyn = np.meshgrid(times, x, y, indexing="ij")
    X_ref = np.stack([xxn.reshape(-1), yyn.reshape(-1), tt.reshape(-1)], axis=1)
    U_ref = fields.reshape(-1, 3)
    case = {
        "seed": int(seed),
        "Re": float(Re),
        "nx": int(nx),
        "nt": int(nt),
        "T": float(T),
        "x": x,
        "y": y,
        "t": times,
        "fields": fields,
        "X_ref": X_ref,
        "U_ref": U_ref,
    }
    np.savez_compressed(path, **case)
    return case


def sample_training_data(case, P_ic=2000, Q_train=5000, seed=0):
    rng = np.random.default_rng(seed)
    Re = float(case["Re"])
    T = float(case["T"])
    X_ic = np.column_stack([rng.random(P_ic) * L, rng.random(P_ic) * L, np.zeros(P_ic)])
    U_ic = np.column_stack(
        [
            periodic_bilinear(case["fields"][0, :, :, k], X_ic[:, :2])[:, 0]
            for k in range(3)
        ]
    )
    X_r = np.column_stack(
        [rng.random(Q_train) * L, rng.random(Q_train) * L, rng.random(Q_train) * T]
    )
    return {
        "Re": np.array([[Re]], dtype=np.float32),
        "X_ic": X_ic.astype(np.float32),
        "U_ic": U_ic.astype(np.float32),
        "X_r": X_r.astype(np.float32),
    }


def relative_l2(pred, target):
    return float(
        np.linalg.norm(pred.reshape(-1) - target.reshape(-1))
        / np.linalg.norm(target.reshape(-1))
    )


if __name__ == "__main__":
    case = generate_case(seed=0, Re=RE, nx=16, nt=3, T=0.02)
    print(
        case["fields"].shape,
        case["X_ref"].shape,
        case["U_ref"].shape,
        float(case["fields"].min()),
        float(case["fields"].max()),
    )
