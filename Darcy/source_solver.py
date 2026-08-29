from pathlib import Path
import time, torch, config
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from data_utils import sample_residual, set_seed
from explicit_derivatives import state_explicit
from model import MLP
from timing import Timer


class SourcePINN:
    def __init__(self, case):
        set_seed(int(case["seed"]))
        self.case, self.device, self.dtype = case, config.DEVICE, config.SOURCE_DTYPE
        self.X_field = torch.tensor(
            case["X_field"], device=self.device, dtype=self.dtype
        )
        self.X_ref = torch.tensor(case["X_ref"], device=self.device, dtype=self.dtype)
        self.truth = torch.tensor(case["S_ref"], device=self.device, dtype=self.dtype)
        self.model = MLP(config.LAYERS, self.X_field).to(self.device, self.dtype)
        self.log = {"outer_step": [], "loss": [], "error": [], "time": []}

    def sample(self, seed):
        d = sample_residual(self.case, config.SOURCE_Q_TRAIN, seed)
        for k in ("X_r", "coeff_r", "coeff_x_r", "coeff_y_r"):
            setattr(self, k, torch.tensor(d[k], device=self.device, dtype=self.dtype))

    def residual(self, theta):
        z = state_explicit(self.model, theta, self.X_r)
        return (
            -(
                self.coeff_x_r * z[:, 1:2]
                + self.coeff_y_r * z[:, 2:3]
                + self.coeff_r * (z[:, 3:4] + z[:, 4:5])
            )
            - 1
        )

    def evaluate(self, theta):
        with torch.no_grad():
            p = self.model_forward(theta, self.X_ref)
            return float(
                (
                    torch.linalg.norm(p - self.truth) / torch.linalg.norm(self.truth)
                ).cpu()
            )

    def model_forward(self, theta, X):
        from torch.func import functional_call
        from model import vector_to_param_dict

        return functional_call(
            self.model, vector_to_param_dict(self.model, theta), (X,)
        )

    def train(self):
        theta = (
            parameters_to_vector(self.model.parameters())
            .detach()
            .clone()
            .requires_grad_(True)
        )
        start = time.perf_counter()
        with Timer() as timer:
            for i in range(config.SOURCE_OUTER_STEPS):
                self.sample(i)
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
                    loss = self.residual(theta).square().mean()
                    loss.backward()
                    return loss

                opt.step(closure)
                with torch.no_grad():
                    loss = float(self.residual(theta).square().mean().cpu())
                err = self.evaluate(theta)
                self.log["outer_step"].append(i + 1)
                self.log["loss"].append(loss)
                self.log["error"].append(err)
                self.log["time"].append(time.perf_counter() - start)
                print(f"outer={i+1:02d} loss={loss:.6e} error={err:.6e}", flush=True)
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
            "derivatives": "explicit_[u,u_x,u_y,u_xx,u_yy]",
            "source_dtype": "float32",
            "train_time_sec": self.train_time,
            "training_config": expected_training_config(),
        }
        torch.save(core, path / "solver_core.pt")
        torch.save(self.log, path / "train_log.pt")
        return core


def expected_training_config():
    return {
        "SOURCE_Q_TRAIN": config.SOURCE_Q_TRAIN,
        "SOURCE_OUTER_STEPS": config.SOURCE_OUTER_STEPS,
        "LBFGS_MAX_ITER": config.LBFGS_MAX_ITER,
        "FIELD_GRID_SIZE": config.FIELD_GRID_SIZE,
        "REFERENCE_GRID_SIZE": config.REFERENCE_GRID_SIZE,
        "resampling": "outer_seed_0_to_29",
    }


def train_source(case, path):
    return SourcePINN(case).train().save(path)
