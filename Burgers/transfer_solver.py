import numpy as np
import torch
from torch.func import jvp, vjp, vmap
from torch.nn.utils import parameters_to_vector

import config
from data_utils import sample_training_data
from explicit_derivatives import burgers_state_explicit, burgers_u_explicit
from model import MLP
from timing import Timer


class TransferSolver:
    def __init__(self, core, device=config.DEVICE):
        self.device = device
        self.source_case = core["case"]
        self.source_error = float(core["log"]["error"][-1])
        self.source_loss = float(core["log"]["loss"][-1])
        xref = torch.tensor(
            core["case"]["X_ref"], device=device, dtype=config.TRANSFER_DTYPE
        )
        self.model = MLP(core["layers"], xref).to(
            device=device, dtype=config.TRANSFER_DTYPE
        )
        self.model.load_state_dict(core["model_state_dict"])
        self.model.eval()
        self.theta = parameters_to_vector(self.model.parameters()).detach()
        self.V = self.S = self.cache = None

    def prepare_points(self, case, seed=0):
        d = sample_training_data(case, config.P_IC, config.P_BC, config.Q_TRAIN, seed)
        self.case = case
        for name in ("X_ic", "s_ic", "X_lbc", "X_ubc", "X_r"):
            dtype = config.TRANSFER_DTYPE
            setattr(self, name, torch.tensor(d[name], device=self.device, dtype=dtype))
        self.X_batch = torch.cat([self.X_ic, self.X_lbc, self.X_ubc, self.X_r])
        self.sizes = [len(self.X_ic), len(self.X_lbc), len(self.X_ubc), len(self.X_r)]

    def response(self, params):
        return burgers_u_explicit(self.model, params, self.X_batch).reshape(-1)

    def extract_basis(self):
        k = config.RANK + config.OVERSAMPLE
        omega = torch.rand(
            self.theta.numel(), k, device=self.device, dtype=self.theta.dtype
        )
        with Timer() as timer:
            jo = vmap(
                lambda d: jvp(self.response, (self.theta,), (d,))[1],
                chunk_size=config.CHUNK_SIZE,
            )(omega.T).T
            q, _ = torch.linalg.qr(jo, mode="reduced")
            _, pullback = vjp(self.response, self.theta)
            jtq = vmap(lambda g: pullback(g)[0], chunk_size=config.CHUNK_SIZE)(q.T).T
            _, s, vh = torch.linalg.svd(jtq.T, full_matrices=False)
            self.S = s[: config.RANK]
            self.V = vh.T[:, : config.RANK] @ torch.diag(1.0 / self.S)
        self.basis_time = timer.seconds

    def build_cache(self):
        with Timer() as timer:
            base = burgers_state_explicit(self.model, self.theta, self.X_batch)
            modes = vmap(
                lambda d: jvp(
                    lambda p: burgers_state_explicit(self.model, p, self.X_batch),
                    (self.theta,),
                    (d,),
                )[1],
                chunk_size=config.CHUNK_SIZE,
            )(self.V.T)
            phi = modes.permute(1, 2, 0)
            b_ic, b_l, b_u, b_r = torch.split(base, self.sizes)
            p_ic, p_l, p_u, p_r = torch.split(phi, self.sizes)
            n_ic, n_bc, n_pde = len(b_ic), len(b_l), len(b_r)
            total = n_ic + 2 * n_bc + n_pde
            self.cache = {
                "base_ic": b_ic[:, :1],
                "phi_ic": p_ic[:, 0, :],
                "base_bc1": b_u[:, :1] - b_l[:, :1],
                "phi_bc1": p_u[:, 0, :] - p_l[:, 0, :],
                "base_bc2": b_u[:, 1:2] - b_l[:, 1:2],
                "phi_bc2": p_u[:, 1, :] - p_l[:, 1, :],
                "base_pde": b_r,
                "phi_pde": p_r,
                "w_ic": (20 * total / n_ic) ** 0.5,
                "w_bc": (total / n_bc) ** 0.5,
                "w_pde": (total / n_pde) ** 0.5,
            }
            xref = torch.tensor(
                self.source_case["X_ref"],
                device=self.device,
                dtype=config.TRANSFER_DTYPE,
            )
            self.cache["base_ref"] = burgers_u_explicit(
                self.model, self.theta, xref
            ).detach()
            self.cache["modes_ref"] = (
                vmap(
                    lambda d: jvp(
                        lambda p: burgers_u_explicit(self.model, p, xref),
                        (self.theta,),
                        (d,),
                    )[1],
                    chunk_size=config.CHUNK_SIZE,
                )(self.V.T)
                .permute(1, 2, 0)
                .detach()
            )
        self.cache_time = timer.seconds

    def set_target(self, case):
        self.case = case
        x = self.X_ic[:, 0].cpu().numpy()
        sic = np.interp(x, case["x"], case["u0"])[:, None]
        self.s_ic = torch.tensor(sic, device=self.device, dtype=config.TRANSFER_DTYPE)

    def residual_and_jacobian(self, alpha):
        c, a = self.cache, alpha[:, None]
        ric = c["base_ic"] - self.s_ic + c["phi_ic"] @ a
        rbc1 = c["base_bc1"] + c["phi_bc1"] @ a
        rbc2 = c["base_bc2"] + c["phi_bc2"] @ a
        state = c["base_pde"] + torch.einsum("ncr,r->nc", c["phi_pde"], alpha)
        u, ux, ut, uxx = state.split(1, dim=1)
        rpde = ut + u * ux - config.NU * uxx
        phi = c["phi_pde"]
        jpde = (
            phi[:, 2, :]
            + phi[:, 0, :] * ux
            + phi[:, 1, :] * u
            - config.NU * phi[:, 3, :]
        )
        residual = torch.cat(
            [c["w_ic"] * ric, c["w_bc"] * rbc1, c["w_bc"] * rbc2, c["w_pde"] * rpde]
        ).reshape(-1)
        jac = torch.cat(
            [
                c["w_ic"] * c["phi_ic"],
                c["w_bc"] * c["phi_bc1"],
                c["w_bc"] * c["phi_bc2"],
                c["w_pde"] * jpde,
            ]
        )
        return residual, jac

    def solve(self, case, need_error=True):
        with Timer() as timer:
            self.set_target(case)
            alpha = torch.zeros(
                config.RANK, device=self.device, dtype=config.TRANSFER_DTYPE
            )
            history = []
            for _ in range(config.NONLINEAR_ITERS):
                residual, jac = self.residual_and_jacobian(alpha)
                history.append(float(residual.square().mean().cpu()))
                alpha += torch.linalg.lstsq(
                    jac, -residual, rcond=config.LSTSQ_RCOND
                ).solution
            residual, _ = self.residual_and_jacobian(alpha)
            history.append(float(residual.square().mean().cpu()))
        result = {
            "res_LST": history[-1],
            "solve_time_sec": timer.seconds,
            "residual_history": history,
            "alpha_norm": float(torch.linalg.norm(alpha).cpu()),
        }
        if need_error:
            truth = torch.tensor(
                case["S_ref"], device=self.device, dtype=config.TRANSFER_DTYPE
            )
            pred = self.cache["base_ref"] + torch.einsum(
                "ncr,r->nc", self.cache["modes_ref"], alpha
            )
            result["error_LST"] = float(
                (torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)).cpu()
            )
        return result
