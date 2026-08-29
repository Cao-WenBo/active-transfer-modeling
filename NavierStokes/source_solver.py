import time
from pathlib import Path

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import config
from data_utils import (
    generate_case,
    relative_l2,
    sample_training_data,
    set_seed,
)
from explicit_derivatives import (
    kolmogorov_pde_residual_explicit,
    kolmogorov_system_explicit,
    kolmogorov_uvw_explicit,
)
from model import Net
from timing import Timer


def expected_training_config():
    return {
        "P_ic": config.SOURCE_P_IC,
        "Q_train": config.SOURCE_Q_TRAIN,
        "outer_steps": config.SOURCE_OUTER_STEPS,
        "lbfgs_max_iter": config.SOURCE_LBFGS_MAX_ITER,
        "resampling": "outer_seed_0_to_29",
        "grf_alpha": config.GRF_ALPHA,
        "grf_tau": config.GRF_TAU,
        "burn_in_time": config.BURN_IN_TIME,
        "coordinate_derivatives": "explicit_minimal_source_residual",
    }


class SourcePINN:
    def __init__(self, case):
        self.case = case
        self.device = config.DEVICE
        self.dtype = config.SOURCE_DTYPE
        set_seed(int(case["seed"]))
        self.X_ref = torch.tensor(case["X_ref"], device=self.device, dtype=self.dtype)
        self.U_ref = torch.tensor(case["U_ref"], device=self.device, dtype=self.dtype)
        self.model = Net(config.LAYERS, self.X_ref).to(
            device=self.device, dtype=self.dtype
        )
        self.log = {
            "outer_step": [],
            "iteration": [],
            "loss": [],
            "pde_loss": [],
            "ic_loss": [],
            "error": [],
            "time": [],
        }
        self.nfev = 0

    def sample(self, seed):
        data = sample_training_data(
            self.case,
            P_ic=config.SOURCE_P_IC,
            Q_train=config.SOURCE_Q_TRAIN,
            seed=seed,
        )
        self.X_ic = torch.tensor(data["X_ic"], device=self.device, dtype=self.dtype)
        self.U_ic = torch.tensor(data["U_ic"], device=self.device, dtype=self.dtype)
        self.X_r = torch.tensor(data["X_r"], device=self.device, dtype=self.dtype)

    def residual(self, theta):
        return kolmogorov_system_explicit(
            self.model, theta, self.X_ic, self.U_ic, self.X_r, config.RE
        )

    def evaluate(self, theta):
        with torch.no_grad():
            prediction = kolmogorov_uvw_explicit(self.model, theta, self.X_ref)
        return relative_l2(prediction.detach().cpu().numpy(), self.case["U_ref"])

    def train(self):
        theta = (
            parameters_to_vector(self.model.parameters())
            .detach()
            .clone()
            .requires_grad_(True)
        )
        start = time.perf_counter()
        with Timer() as timer:
            for outer in range(config.SOURCE_OUTER_STEPS):
                self.sample(outer)
                optimizer = torch.optim.LBFGS(
                    [theta],
                    max_iter=config.SOURCE_LBFGS_MAX_ITER,
                    history_size=config.SOURCE_LBFGS_MAX_ITER,
                    tolerance_grad=1e-10,
                    tolerance_change=1e-12,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    optimizer.zero_grad()
                    self.nfev += 1
                    loss = self.residual(theta).square().mean()
                    loss.backward()
                    return loss

                optimizer.step(closure)
                with torch.no_grad():
                    residual = self.residual(theta)
                    loss = float(residual.square().mean().cpu())
                    pde = float(
                        kolmogorov_pde_residual_explicit(
                            self.model, theta, self.X_r, config.RE
                        )
                        .square()
                        .mean()
                        .cpu()
                    )
                    ic = float(
                        (
                            kolmogorov_uvw_explicit(self.model, theta, self.X_ic)
                            - self.U_ic
                        )
                        .square()
                        .mean()
                        .cpu()
                    )
                error = self.evaluate(theta)
                self.log["outer_step"].append(outer + 1)
                self.log["iteration"].append(self.nfev)
                self.log["loss"].append(loss)
                self.log["pde_loss"].append(pde)
                self.log["ic_loss"].append(ic)
                self.log["error"].append(error)
                self.log["time"].append(time.perf_counter() - start)
                print(
                    f"source outer={outer + 1:02d} loss={loss:.6e} "
                    f"pde={pde:.6e} ic={ic:.6e} error={error:.6e}",
                    flush=True,
                )
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
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "source_dtype": "float32",
            "train_time_sec": self.train_time,
            "training_config": expected_training_config(),
        }
        torch.save(core, path / "solver_core.pt")
        torch.save(self.log, path / "train_log.pt")
        return core


def train_source(seed, path):
    case = generate_case(
        seed=int(seed),
        Re=config.RE,
        nx=config.NX,
        nt=config.SOURCE_NT,
        T=config.TIME_INTERVAL,
        spinup=config.BURN_IN_TIME,
        device=config.DEVICE,
    )
    return SourcePINN(case).train().save(path)
