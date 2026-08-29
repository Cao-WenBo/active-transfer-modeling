import time

import numpy as np
import torch

import config
from torch.func import jvp, vjp, vmap
from torch.nn.utils import parameters_to_vector
from data_utils import eval_coeff_and_grad, sample_residual
from explicit_derivatives import state_explicit
from model import MLP
from timing import Timer


class TransferSolver:
    def __init__(self, core):
        self.device, self.dtype = config.DEVICE, config.TRANSFER_DTYPE
        self.source_case = core["case"]
        self.source_error = float(core["log"]["error"][-1])
        self.source_loss = float(core["log"]["loss"][-1])
        x_field = torch.tensor(
            self.source_case["X_field"], device=self.device, dtype=self.dtype
        )
        self.model = MLP(core["layers"], x_field).to(self.device, self.dtype)
        self.model.load_state_dict(core["model_state_dict"])
        self.model.eval()
        self.theta = parameters_to_vector(self.model.parameters()).detach()

    def prepare(self):
        d = sample_residual(
            self.source_case, config.TRANSFER_Q_TRAIN, config.SHARED_SAMPLE_SEED
        )
        self.X_r = torch.tensor(d["X_r"], device=self.device, dtype=self.dtype)
        for k in ("coeff_r", "coeff_x_r", "coeff_y_r"):
            setattr(self, k, torch.tensor(d[k], device=self.device, dtype=self.dtype))

    def response(self, p):
        """Network-output response used to construct the transferable subspace."""
        return self._forward(p, self.X_r).reshape(-1)

    def extract_basis(self):
        omega = torch.randn(
            self.theta.numel(),
            config.RANK + config.OVERSAMPLE,
            device=self.device,
            dtype=self.dtype,
        )
        with Timer() as t:
            jo = vmap(
                lambda d: jvp(self.response, (self.theta,), (d,))[1],
                chunk_size=config.CHUNK_SIZE,
            )(omega.T).T
            q, _ = torch.linalg.qr(jo, mode="reduced")
            _, pb = vjp(self.response, self.theta)
            jtq = vmap(lambda g: pb(g)[0], chunk_size=config.CHUNK_SIZE)(q.T).T
            _, s, vh = torch.linalg.svd(jtq.T, full_matrices=False)
            self.S = s[: config.RANK]
            self.V = vh.T[:, : config.RANK] @ torch.diag(1 / self.S)
        self.basis_time = t.seconds

    def build_cache(self):
        with Timer() as t:
            self.base_r = state_explicit(self.model, self.theta, self.X_r)
            self.modes_r = vmap(
                lambda d: jvp(
                    lambda p: state_explicit(self.model, p, self.X_r),
                    (self.theta,),
                    (d,),
                )[1],
                chunk_size=config.CHUNK_SIZE,
            )(self.V.T).permute(1, 2, 0)
        self.cache_time = t.seconds

    def _forward(self, p, x):
        from torch.func import functional_call
        from model import vector_to_param_dict

        return functional_call(self.model, vector_to_param_dict(self.model, p), (x,))

    def solve(self, case, need_error=True):
        start = time.perf_counter()
        a, ax, ay = eval_coeff_and_grad(case, self.X_r.detach().cpu().numpy())
        a = torch.tensor(a, device=self.device, dtype=self.dtype)
        ax = torch.tensor(ax, device=self.device, dtype=self.dtype)
        ay = torch.tensor(ay, device=self.device, dtype=self.dtype)
        b = self.base_r
        p = self.modes_r
        f = (
            -(ax * b[:, 1:2] + ay * b[:, 2:3] + a * (b[:, 3:4] + b[:, 4:5])) - 1
        ).reshape(-1)
        A = -(ax * p[:, 1, :] + ay * p[:, 2, :] + a * (p[:, 3, :] + p[:, 4, :]))
        alpha = torch.linalg.lstsq(A, -f, rcond=config.LSTSQ_RCOND).solution
        r = f + A @ alpha
        loss = float(r.square().mean().cpu())
        solve_time = time.perf_counter() - start
        out = {"res_LST": loss, "solve_time_sec": solve_time}
        if need_error:
            delta_theta = self.V @ alpha
            truth = np.asarray(case["S_ref"], dtype=np.float64).reshape(-1)
            points = np.asarray(case["X_ref"], dtype=np.float64)
            numerator = 0.0
            denominator = float(np.dot(truth, truth))
            for begin in range(0, len(points), config.REFERENCE_JVP_CHUNK):
                end = min(begin + config.REFERENCE_JVP_CHUNK, len(points))
                x = torch.tensor(
                    points[begin:end], device=self.device, dtype=self.dtype
                )
                with torch.no_grad():
                    base = self.model(x).reshape(-1)
                tangent = jvp(
                    lambda p: self._forward(p, x).reshape(-1),
                    (self.theta,),
                    (delta_theta,),
                )[1]
                prediction = (base + tangent).detach().cpu().numpy()
                difference = prediction - truth[begin:end]
                numerator += float(np.dot(difference, difference))
            out["error_LST"] = float(np.sqrt(numerator / denominator))
        return out
