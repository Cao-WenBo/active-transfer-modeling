from pathlib import Path
import time
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import config
from data_utils import sample_training_data, set_seed
from explicit_derivatives import state_explicit
from model import MLP
from timing import Timer


class SourcePINN:
    def __init__(self, case):
        set_seed(int(case["seed"]))
        self.case = case
        self.device, self.dtype = config.DEVICE, config.SOURCE_DTYPE
        xref = torch.tensor(case["X_ref"], device=self.device, dtype=self.dtype)
        self.truth = torch.tensor(case["S_ref"], device=self.device, dtype=self.dtype)
        self.model = MLP(config.LAYERS, xref).to(self.device, self.dtype)
        self.X_ref = xref
        self.log = {"outer_step": [], "loss": [], "error": [], "time": []}

    def sample(self):
        d = sample_training_data(self.case)
        for key in ("X_bc", "s_bc", "X_r", "rhs_r"):
            setattr(
                self, key, torch.tensor(d[key], device=self.device, dtype=self.dtype)
            )
        self.X_batch = torch.cat([self.X_bc, self.X_r])
        self.sizes = [len(self.X_bc), len(self.X_r)]

    def residual(self, theta):
        state = state_explicit(self.model, theta, self.X_batch)
        bc, pde = torch.split(state, self.sizes)
        rbc = bc[:, :1] - self.s_bc
        rpde = pde[:, 1:2] - self.rhs_r
        total = len(rbc) + len(rpde)
        return torch.cat(
            [
                (total / len(rbc)) ** 0.5 * rbc,
                (total / len(rpde)) ** 0.5 * rpde,
            ]
        )

    def evaluate(self, theta):
        with torch.no_grad():
            pred = state_explicit(self.model, theta, self.X_ref)[:, :1]
            return float(
                (
                    torch.linalg.norm(pred - self.truth) / torch.linalg.norm(self.truth)
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
            for outer in range(config.SOURCE_OUTER_STEPS):
                optimizer = torch.optim.LBFGS(
                    [theta],
                    max_iter=config.LBFGS_MAX_ITER,
                    history_size=config.LBFGS_MAX_ITER,
                    tolerance_grad=1e-10,
                    tolerance_change=1e-12,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    optimizer.zero_grad()
                    loss = self.residual(theta).square().mean()
                    loss.backward()
                    return loss

                optimizer.step(closure)
                with torch.no_grad():
                    loss = float(self.residual(theta).square().mean().cpu())
                self.log["outer_step"].append(outer + 1)
                self.log["loss"].append(loss)
                self.log["error"].append(self.evaluate(theta))
                self.log["time"].append(time.perf_counter() - start)
        vector_to_parameters(theta.detach(), self.model.parameters())
        self.train_time = timer.seconds
        return self

    def save(self, run_dir):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "layers": config.LAYERS,
            "case": self.case,
            "log": self.log,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            },
            "derivatives": "explicit_batched_[s,s_x]",
            "source_dtype": "float32",
            "train_time_sec": self.train_time,
            "training_config": {
                "P_TRAIN": config.P_TRAIN,
                "Q_TRAIN": config.Q_TRAIN,
                "SOURCE_OUTER_STEPS": config.SOURCE_OUTER_STEPS,
                "LBFGS_MAX_ITER": config.LBFGS_MAX_ITER,
                "LBFGS_RESAMPLE": config.LBFGS_RESAMPLE,
            },
        }
        torch.save(payload, run_dir / "solver_core.pt")
        torch.save(self.log, run_dir / "train_log.pt")
        return payload


def train_source(case, run_dir):
    return SourcePINN(case).train().save(run_dir)
