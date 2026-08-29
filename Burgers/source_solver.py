from pathlib import Path
import time
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import config
from data_utils import sample_training_data, set_seed
from explicit_derivatives import burgers_state_explicit, burgers_u_explicit
from model import MLP
from timing import Timer


class SourcePINN:
    def __init__(self, case, device=config.DEVICE, dtype=config.TRAIN_DTYPE):
        self.case, self.device, self.dtype = case, device, dtype
        self.X_ref = torch.tensor(case["X_ref"], device=device, dtype=dtype)
        self.s_ref = torch.tensor(case["S_ref"], device=device, dtype=dtype)
        self.model = MLP(config.LAYERS, self.X_ref).to(device=device, dtype=dtype)
        self.log = {"loss": [], "error": [], "iteration": [], "time": []}

    def sample(self, seed=0):
        d = sample_training_data(
            self.case, config.P_IC, config.P_BC, config.Q_TRAIN, seed
        )
        for name in ("X_ic", "s_ic", "X_lbc", "X_ubc", "X_r"):
            setattr(
                self, name, torch.tensor(d[name], device=self.device, dtype=self.dtype)
            )
        self.X_batch = torch.cat([self.X_ic, self.X_lbc, self.X_ubc, self.X_r], dim=0)
        self.split_sizes = [
            len(self.X_ic),
            len(self.X_lbc),
            len(self.X_ubc),
            len(self.X_r),
        ]

    def residual(self, theta):
        state = burgers_state_explicit(self.model, theta, self.X_batch)
        state_ic, left, right, state_r = torch.split(state, self.split_sizes)
        ic = state_ic[:, :1] - self.s_ic
        bc_u = right[:, :1] - left[:, :1]
        bc_ux = right[:, 1:2] - left[:, 1:2]
        u, ux, ut, uxx = state_r.split(1, dim=1)
        pde = ut + u * ux - config.NU * uxx
        n_ic, n_bc, n_pde = len(ic), len(bc_u), len(pde)
        total = n_ic + 2 * n_bc + n_pde
        return torch.cat(
            [
                (20 * total / n_ic) ** 0.5 * ic,
                (total / n_bc) ** 0.5 * bc_u,
                (total / n_bc) ** 0.5 * bc_ux,
                (total / n_pde) ** 0.5 * pde,
            ]
        )

    def evaluate(self, theta):
        with torch.no_grad():
            pred = burgers_u_explicit(self.model, theta, self.X_ref)
            return float(
                (
                    torch.linalg.norm(pred - self.s_ref) / torch.linalg.norm(self.s_ref)
                ).cpu()
            )

    def train(self, collocation_seed=0):
        self.sample(collocation_seed)
        theta = (
            parameters_to_vector(self.model.parameters())
            .detach()
            .clone()
            .requires_grad_(True)
        )
        nfev, start = 0, time.perf_counter()
        optimizer = torch.optim.LBFGS(
            [theta],
            max_iter=config.LBFGS_MAX_ITER,
            history_size=config.LBFGS_MAX_ITER,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal nfev
            optimizer.zero_grad()
            loss = self.residual(theta).square().mean()
            loss.backward()
            nfev += 1
            return loss

        with Timer() as timer:
            outer_step = 0
            while nfev < config.SOURCE_MIN_ITERATIONS:
                if config.LBFGS_RESAMPLE and outer_step > 0:
                    self.sample(collocation_seed + outer_step)
                loss = optimizer.step(closure)
                self.log["loss"].append(float(loss.detach().cpu()))
                self.log["error"].append(self.evaluate(theta))
                self.log["iteration"].append(nfev)
                self.log["time"].append(time.perf_counter() - start)
                outer_step += 1
        vector_to_parameters(theta.detach(), self.model.parameters())
        self.train_time = timer.seconds
        return self

    def save(self, run_dir, model_seed=None, collocation_seed=0):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "layers": config.LAYERS,
            "case": self.case,
            "log": self.log,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            },
            "derivatives": "explicit_minimal_[u,ux,ut,uxx]",
            "train_dtype": "float32",
            "train_time_sec": self.train_time,
            "training_config": {
                "P_IC": config.P_IC,
                "P_BC": config.P_BC,
                "Q_TRAIN": config.Q_TRAIN,
                "SOURCE_MIN_ITERATIONS": config.SOURCE_MIN_ITERATIONS,
                "LBFGS_OUTER_STEPS": config.LBFGS_OUTER_STEPS,
                "LBFGS_MAX_ITER": config.LBFGS_MAX_ITER,
                "LBFGS_RESAMPLE": config.LBFGS_RESAMPLE,
                "seed": int(collocation_seed),
            },
        }
        if model_seed is not None:
            payload["random_seeds"] = {
                "model_initialization": int(model_seed),
                "source_collocation": int(collocation_seed),
            }
        torch.save(payload, run_dir / "solver_core.pt")
        torch.save(self.log, run_dir / "train_log.pt")
        return payload


def train_source(case, run_dir, model_seed=None, collocation_seed=0):
    path = Path(run_dir) / "solver_core.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    if model_seed is not None:
        set_seed(model_seed)
    return (
        SourcePINN(case)
        .train(collocation_seed=collocation_seed)
        .save(
            run_dir,
            model_seed=model_seed,
            collocation_seed=collocation_seed,
        )
    )
