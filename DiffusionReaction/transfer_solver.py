import time, torch
from torch.func import jvp, vjp, vmap
from torch.nn.utils import parameters_to_vector
import config
from data_utils import sample_training_data
from explicit_derivatives import state_explicit
from model import MLP
from timing import Timer


class TransferSolver:
    def __init__(self, core):
        self.device = config.DEVICE
        self.dtype = config.TRANSFER_DTYPE
        self.source_case = core["case"]
        self.source_error = float(core["log"]["error"][-1])
        self.source_loss = float(core["log"]["loss"][-1])
        x = torch.tensor(
            self.source_case["X_ref"], device=self.device, dtype=self.dtype
        )
        self.model = MLP(core["layers"], x).to(self.device, self.dtype)
        self.model.load_state_dict(core["model_state_dict"])
        self.model.eval()
        self.theta = parameters_to_vector(self.model.parameters()).detach()

    def prepare(self):
        d = sample_training_data(self.source_case, 0)
        for k in ("X_bc", "s_bc", "X_r"):
            setattr(self, k, torch.tensor(d[k], device=self.device, dtype=self.dtype))
        self.X_batch = torch.cat([self.X_bc, self.X_r])
        self.sizes = [len(self.X_bc), len(self.X_r)]

    def response(self, p):
        return state_explicit(self.model, p, self.X_batch)[:, :1].reshape(-1)

    def extract_basis(self):
        omega = torch.rand(
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
            base = state_explicit(self.model, self.theta, self.X_batch)
            phi = vmap(
                lambda d: jvp(
                    lambda p: state_explicit(self.model, p, self.X_batch),
                    (self.theta,),
                    (d,),
                )[1],
                chunk_size=config.CHUNK_SIZE,
            )(self.V.T).permute(1, 2, 0)
            bbc, br = torch.split(base, self.sizes)
            pbc, pr = torch.split(phi, self.sizes)
            n = len(bbc) + len(br)
            self.cache = {
                "bbc": bbc[:, :1],
                "pbc": pbc[:, 0, :],
                "br": br,
                "pr": pr,
                "wbc": (n / len(bbc)) ** 0.5,
                "wpde": (n / len(br)) ** 0.5,
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
        self.cache_time = t.seconds

    def solve(self, case, need_error=True):
        start = time.perf_counter()
        forcing = torch.tensor(
            sample_training_data(case, 0)["forcing_r"],
            device=self.device,
            dtype=self.dtype,
        )
        c = self.cache
        alpha = torch.zeros(config.RANK, device=self.device, dtype=self.dtype)
        history = []
        for _ in range(config.NONLINEAR_ITERS):
            state = c["br"] + torch.einsum("ncr,r->nc", c["pr"], alpha)
            u, ut, uxx = state.split(1, 1)
            rb = c["bbc"] - self.s_bc + c["pbc"] @ alpha[:, None]
            rp = ut - 0.01 * uxx - 0.01 * u.square() - forcing
            residual = torch.cat([c["wbc"] * rb, c["wpde"] * rp]).reshape(-1)
            pr = c["pr"]
            jp = pr[:, 1, :] - 0.01 * pr[:, 2, :] - 0.02 * u * pr[:, 0, :]
            jac = torch.cat([c["wbc"] * c["pbc"], c["wpde"] * jp])
            history.append(float(residual.square().mean().cpu()))
            alpha += torch.linalg.lstsq(
                jac, -residual, rcond=config.LSTSQ_RCOND
            ).solution
        state = c["br"] + torch.einsum("ncr,r->nc", c["pr"], alpha)
        u, ut, uxx = state.split(1, 1)
        residual = torch.cat(
            [
                c["wbc"] * (c["bbc"] - self.s_bc + c["pbc"] @ alpha[:, None]),
                c["wpde"] * (ut - 0.01 * uxx - 0.01 * u.square() - forcing),
            ]
        ).reshape(-1)
        final_loss = float(residual.square().mean().cpu())
        result = {
            "res_LST": final_loss,
            "residual_history": history + [final_loss],
            "solve_time_sec": time.perf_counter() - start,
        }
        if need_error:
            truth = torch.tensor(case["S_ref"], device=self.device, dtype=self.dtype)
            pred = c["base_ref"] + torch.einsum("ncr,r->nc", c["modes_ref"], alpha)
            result["error_LST"] = float(
                (torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)).cpu()
            )
        return result
