import time
import torch
from torch.func import jvp, vjp, vmap
from torch.nn.utils import parameters_to_vector

import config
from data_utils import sample_training_data
from explicit_derivatives import state_explicit
from model import MLP
from timing import Timer


class TransferSolver:
    def __init__(self, core):
        self.device, self.dtype = config.DEVICE, config.TRANSFER_DTYPE
        self.source_case = core["case"]
        self.source_error = float(core["log"]["error"][-1])
        self.source_loss = float(core["log"]["loss"][-1])
        xref = torch.tensor(
            self.source_case["X_ref"], device=self.device, dtype=self.dtype
        )
        self.model = MLP(core["layers"], xref).to(self.device, self.dtype)
        self.model.load_state_dict(core["model_state_dict"])
        self.model.eval()
        self.theta = parameters_to_vector(self.model.parameters()).detach()

    def prepare(self):
        d = sample_training_data(self.source_case)
        self.X_bc = torch.tensor(d["X_bc"], device=self.device, dtype=self.dtype)
        self.X_r = torch.tensor(d["X_r"], device=self.device, dtype=self.dtype)
        self.s_bc = torch.tensor(d["s_bc"], device=self.device, dtype=self.dtype)
        self.X_batch = torch.cat([self.X_bc, self.X_r])
        self.sizes = [len(self.X_bc), len(self.X_r)]

    def response(self, params):
        return state_explicit(self.model, params, self.X_batch)[:, :1].reshape(-1)

    def extract_basis(self):
        omega = torch.rand(
            self.theta.numel(),
            config.RANK + config.OVERSAMPLE,
            device=self.device,
            dtype=self.dtype,
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
            base = state_explicit(self.model, self.theta, self.X_batch)
            modes = vmap(
                lambda d: jvp(
                    lambda p: state_explicit(self.model, p, self.X_batch),
                    (self.theta,),
                    (d,),
                )[1],
                chunk_size=config.CHUNK_SIZE,
            )(self.V.T).permute(1, 2, 0)
            base_bc, base_r = torch.split(base, self.sizes)
            phi_bc, phi_r = torch.split(modes, self.sizes)
            total = len(base_bc) + len(base_r)
            self.cache = {
                "base_bc": base_bc[:, :1],
                "phi_bc": phi_bc[:, 0, :],
                "base_dx": base_r[:, 1:2],
                "phi_dx": phi_r[:, 1, :],
                "w_bc": (total / len(base_bc)) ** 0.5,
                "w_pde": (total / len(base_r)) ** 0.5,
            }
            xref = torch.tensor(
                self.source_case["X_ref"], device=self.device, dtype=self.dtype
            )
            self.cache["base_ref"] = state_explicit(self.model, self.theta, xref)[
                :, :1
            ].detach()
            self.cache["modes_ref"] = (
                vmap(
                    lambda d: jvp(
                        lambda p: state_explicit(self.model, p, xref)[:, :1],
                        (self.theta,),
                        (d,),
                    )[1],
                    chunk_size=config.CHUNK_SIZE,
                )(self.V.T)
                .permute(1, 2, 0)
                .detach()
            )
        self.cache_time = timer.seconds

    def solve(self, case, need_error=True):
        start = time.perf_counter()
        d = sample_training_data(case)
        rhs = torch.tensor(d["rhs_r"], device=self.device, dtype=self.dtype)
        c = self.cache
        f0 = torch.cat(
            [
                c["w_bc"] * (c["base_bc"] - self.s_bc),
                c["w_pde"] * (c["base_dx"] - rhs),
            ]
        ).reshape(-1)
        design = torch.cat(
            [
                c["w_bc"] * c["phi_bc"],
                c["w_pde"] * c["phi_dx"],
            ]
        )
        alpha = torch.linalg.lstsq(design, -f0, rcond=config.LSTSQ_RCOND).solution
        residual_loss = float((f0 + design @ alpha).square().mean().cpu())
        solve_time = time.perf_counter() - start
        result = {
            "res_LST": residual_loss,
            "solve_time_sec": solve_time,
            "alpha_norm": float(torch.linalg.norm(alpha).cpu()),
        }
        if need_error:
            truth = torch.tensor(case["S_ref"], device=self.device, dtype=self.dtype)
            pred = self.cache["base_ref"] + torch.einsum(
                "ncr,r->nc", self.cache["modes_ref"], alpha
            )
            result["error_LST"] = float(
                (torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)).cpu()
            )
        return result
