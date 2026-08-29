import time, torch, config
from torch.func import jvp, vjp, vmap
from torch.nn.utils import parameters_to_vector
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
        x = torch.tensor(
            self.source_case["X_ref"], device=self.device, dtype=self.dtype
        )
        self.source_loss = float(core["log"]["loss"][-1])
        self.model = MLP(core["layers"], x).to(self.device, self.dtype)
        self.model.load_state_dict(core["model_state_dict"])
        self.model.eval()
        self.theta = parameters_to_vector(self.model.parameters()).detach()

    def prepare(self):
        d = sample_training_data(self.source_case)
        for k in ("X_bc", "s_bc", "X_r"):
            setattr(self, k, torch.tensor(d[k], device=self.device, dtype=self.dtype))
        self.X_batch = torch.cat([self.X_bc, self.X_r])
        self.sizes = [len(self.X_bc), len(self.X_r)]

    def response(self, p):
        return state_explicit(self.model, p, self.X_batch)[:, :1].reshape(-1)

    def extract_basis(self):
        o = torch.rand(
            self.theta.numel(),
            config.RANK + config.OVERSAMPLE,
            device=self.device,
            dtype=self.dtype,
        )
        with Timer() as t:
            jo = vmap(
                lambda d: jvp(self.response, (self.theta,), (d,))[1],
                chunk_size=config.CHUNK_SIZE,
            )(o.T).T
            q, _ = torch.linalg.qr(jo, mode="reduced")
            _, pb = vjp(self.response, self.theta)
            jtq = vmap(lambda g: pb(g)[0], chunk_size=config.CHUNK_SIZE)(q.T).T
            _, s, vh = torch.linalg.svd(jtq.T, full_matrices=False)
            self.S = s[: config.RANK]
            self.V = vh.T[:, : config.RANK] @ torch.diag(1 / self.S)
        self.basis_time = t.seconds

    def build_cache(self):
        with Timer() as t:
            b = state_explicit(self.model, self.theta, self.X_batch)
            p = vmap(
                lambda d: jvp(
                    lambda q: state_explicit(self.model, q, self.X_batch),
                    (self.theta,),
                    (d,),
                )[1],
                chunk_size=config.CHUNK_SIZE,
            )(self.V.T).permute(1, 2, 0)
            bb, br = torch.split(b, self.sizes)
            pb, pr = torch.split(p, self.sizes)
            n = len(bb) + len(br)
            self.cache = {
                "bb": bb[:, :1],
                "pb": pb[:, 0, :],
                "br": br,
                "pr": pr,
                "wb": (100 * n / len(bb)) ** 0.5,
                "wp": (n / len(br)) ** 0.5,
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
                        lambda q: state_explicit(self.model, q, xref)[:, :1],
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
        ux = torch.tensor(
            sample_training_data(case)["ux_r"], device=self.device, dtype=self.dtype
        )
        c = self.cache
        f = torch.cat(
            [
                c["wb"] * (c["bb"] - self.s_bc),
                c["wp"] * (c["br"][:, 2:3] + ux * c["br"][:, 1:2]),
            ]
        ).reshape(-1)
        a = torch.cat(
            [c["wb"] * c["pb"], c["wp"] * (c["pr"][:, 2, :] + ux * c["pr"][:, 1, :])]
        )
        alpha = torch.linalg.lstsq(a, -f, rcond=config.LSTSQ_RCOND).solution
        r = f + a @ alpha
        loss = float(r.square().mean().cpu())
        out = {"res_LST": loss, "solve_time_sec": time.perf_counter() - start}
        if need_error:
            truth = torch.tensor(case["S_ref"], device=self.device, dtype=self.dtype)
            pred = c["base_ref"] + torch.einsum("ncr,r->nc", c["modes_ref"], alpha)
            out["error_LST"] = float(
                (torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)).cpu()
            )
        return out
