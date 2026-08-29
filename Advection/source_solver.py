from pathlib import Path
import time, torch, config
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from data_utils import sample_training_data, set_seed
from explicit_derivatives import state_explicit
from model import MLP
from timing import Timer


class SourcePINN:
    def __init__(self, case):
        set_seed(int(case["seed"]))
        self.case = case
        self.device = config.DEVICE
        self.dtype = config.SOURCE_DTYPE
        self.X_ref = torch.tensor(case["X_ref"], device=self.device, dtype=self.dtype)
        self.truth = torch.tensor(case["S_ref"], device=self.device, dtype=self.dtype)
        self.model = MLP(config.LAYERS, self.X_ref).to(self.device, self.dtype)
        self.log = {"outer_step": [], "loss": [], "error": [], "time": []}

    def sample(self):
        d = sample_training_data(self.case)
        for k in ("X_bc", "s_bc", "X_r", "ux_r"):
            setattr(self, k, torch.tensor(d[k], device=self.device, dtype=self.dtype))
        self.X_batch = torch.cat([self.X_bc, self.X_r])
        self.sizes = [len(self.X_bc), len(self.X_r)]

    def residual(self, theta):
        z = state_explicit(self.model, theta, self.X_batch)
        bc, r = torch.split(z, self.sizes)
        rb = bc[:, :1] - self.s_bc
        rp = r[:, 2:3] + self.ux_r * r[:, 1:2]
        n = len(rb) + len(rp)
        return torch.cat([(100 * n / len(rb)) ** 0.5 * rb, (n / len(rp)) ** 0.5 * rp])

    def evaluate(self, theta):
        with torch.no_grad():
            p = state_explicit(self.model, theta, self.X_ref)[:, :1]
            return float(
                (
                    torch.linalg.norm(p - self.truth) / torch.linalg.norm(self.truth)
                ).cpu()
            )

    def train(self):
        self.sample()
        theta = (
            parameters_to_vector(self.model.parameters())
            .detach()
            .clone()
            .requires_grad_(True)
        )
        start = time.perf_counter()
        with Timer() as timer:
            for i in range(config.SOURCE_OUTER_STEPS):
                opt = torch.optim.LBFGS(
                    [theta],
                    max_iter=config.LBFGS_MAX_ITER,
                    history_size=config.LBFGS_MAX_ITER,
                    tolerance_grad=1e-10,
                    tolerance_change=1e-12,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    opt.zero_grad()
                    l = self.residual(theta).square().mean()
                    l.backward()
                    return l

                opt.step(closure)
                with torch.no_grad():
                    l = float(self.residual(theta).square().mean().cpu())
                self.log["outer_step"].append(i + 1)
                self.log["loss"].append(l)
                self.log["error"].append(self.evaluate(theta))
                self.log["time"].append(time.perf_counter() - start)
        vector_to_parameters(theta.detach(), self.model.parameters())
        self.train_time = timer.seconds
        return self

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        core = {
            "layers": config.LAYERS,
            "case": self.case,
            "log": self.log,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            },
            "derivatives": "explicit_batched_[s,s_x,s_t]",
            "source_dtype": "float32",
            "train_time_sec": self.train_time,
            "training_config": {
                "P_TRAIN": config.P_TRAIN,
                "Q_TRAIN": config.Q_TRAIN,
                "SOURCE_OUTER_STEPS": config.SOURCE_OUTER_STEPS,
                "LBFGS_MAX_ITER": config.LBFGS_MAX_ITER,
                "sampling_seed": 0,
            },
        }
        torch.save(core, path / "solver_core.pt")
        torch.save(self.log, path / "train_log.pt")
        return core


def train_source(case, path):
    return SourcePINN(case).train().save(path)
