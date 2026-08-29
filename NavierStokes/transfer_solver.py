import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp, vjp, vmap
from torch.nn.utils import parameters_to_vector

import config
from data_utils import FORCE_N, L, set_seed
from explicit_derivatives import (
    kolmogorov_state_minimal_explicit,
    kolmogorov_uvw_explicit,
)
from model import Net
from timing import Timer


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TransferSolver:
    def __init__(self, core, source_seed):
        self.device = config.DEVICE
        self.dtype = config.TRANSFER_DTYPE
        self.source_seed = int(source_seed)
        self.source_case = core["case"]
        self.source_error = float(core["log"]["error"][-1])
        self.source_loss = float(core["log"]["loss"][-1])
        reference = torch.tensor(
            self.source_case["X_ref"], device=self.device, dtype=self.dtype
        )
        self.model = Net(core["layers"], reference).to(
            device=self.device, dtype=self.dtype
        )
        self.model.load_state_dict(core["model_state_dict"])
        self.model.eval()
        self.theta = parameters_to_vector(self.model.parameters()).detach()
        self.V = None
        self.S = None
        self.basis_time = 0.0
        self.cache_time = 0.0
        self.basis_reused = False
        self.cache_reused = False

    @property
    def cache_dir(self):
        return config.CACHE_ROOT / f"seed{self.source_seed:05d}"

    @property
    def basis_path(self):
        return self.cache_dir / "basis.pt"

    @property
    def mode_cache_path(self):
        return self.cache_dir / "time_marching_cache.pt"

    def basis_points(self):
        rng = np.random.default_rng(config.BASIS_POINT_SEED)
        points = np.column_stack(
            [
                rng.random(config.P_BASIS) * 2.0 * np.pi,
                rng.random(config.P_BASIS) * 2.0 * np.pi,
                rng.random(config.P_BASIS) * config.TIME_INTERVAL,
            ]
        )
        return torch.tensor(points, device=self.device, dtype=self.dtype)

    def response(self, theta):
        return kolmogorov_uvw_explicit(self.model, theta, self.X_basis).reshape(-1)

    def extract_or_load_basis(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.basis_path.exists():
            payload = torch.load(
                self.basis_path, map_location="cpu", weights_only=False
            )
            if int(payload["V"].shape[1]) < config.RANK:
                raise RuntimeError(f"basis rank is too small: {self.basis_path}")
            self.V = payload["V"][:, : config.RANK].to(
                device=self.device, dtype=self.dtype
            )
            self.S = payload["S"][: config.RANK].to(
                device=self.device, dtype=self.dtype
            )
            self.X_basis = payload["X_basis"].to(device=self.device, dtype=self.dtype)
            self.basis_time = float(
                payload.get("basis_time_sec", payload.get("basis_time", 0.0))
            )
            self.basis_reused = True
            return

        self.X_basis = self.basis_points()
        set_seed(config.BASIS_SKETCH_SEED_BASE + self.source_seed)
        probes = torch.randn(
            self.theta.numel(),
            config.RANK + config.OVERSAMPLE,
            device=self.device,
            dtype=self.dtype,
        )
        synchronize()
        with Timer() as timer:
            response_probe = vmap(
                lambda direction: jvp(self.response, (self.theta,), (direction,))[1],
                chunk_size=config.BASIS_JVP_CHUNK,
            )(probes.T).T
            q, _ = torch.linalg.qr(response_probe, mode="reduced")
            _, pullback = vjp(self.response, self.theta)
            jtq = vmap(
                lambda vector: pullback(vector)[0],
                chunk_size=config.BASIS_JVP_CHUNK,
            )(q.T).T
            _, singular, vh = torch.linalg.svd(jtq.T, full_matrices=False)
            self.S = singular[: config.RANK]
            self.V = vh.T[:, : config.RANK] @ torch.diag(1.0 / self.S)
            synchronize()
        self.basis_time = timer.seconds
        torch.save(
            {
                "V": self.V.detach().cpu(),
                "S": self.S.detach().cpu(),
                "X_basis": self.X_basis.detach().cpu(),
                "rank": config.RANK,
                "oversample": config.OVERSAMPLE,
                "P_basis": config.P_BASIS,
                "basis_point_seed": config.BASIS_POINT_SEED,
                "basis_sketch_seed": (config.BASIS_SKETCH_SEED_BASE + self.source_seed),
                "response": "[u,v,w]",
                "basis_time_sec": self.basis_time,
            },
            self.basis_path,
        )

    def spatial_points(self):
        nodes = np.linspace(0.0, L, config.NX, endpoint=False, dtype=np.float32)
        xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
        return np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    def points_at_time(self, xy, value):
        return torch.tensor(
            np.column_stack(
                [
                    xy,
                    np.full(xy.shape[0], value, dtype=np.float32),
                ]
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def build_or_load_cache(self):
        if self.mode_cache_path.exists():
            payload = torch.load(
                self.mode_cache_path, map_location="cpu", weights_only=False
            )
            self.xy = payload["xy"].numpy().astype(np.float32)
            self.times = payload["times"].numpy().astype(np.float32)
            self.base_state_all = payload["base_state_all"].to(
                device=self.device, dtype=self.dtype
            )
            self.modes_state = payload["modes_state"][: config.RANK].to(
                device=self.device, dtype=self.dtype
            )
            self.cache_time = float(payload.get("cache_time_sec", 0.0))
            self.cache_reused = True
            return

        self.xy = self.spatial_points()
        self.times = np.linspace(
            0.0,
            config.TIME_INTERVAL,
            config.NT,
            dtype=np.float32,
        )
        synchronize()
        with Timer() as timer:
            base_states = []
            for value in self.times:
                points = self.points_at_time(self.xy, value)
                base_states.append(
                    kolmogorov_state_minimal_explicit(self.model, self.theta, points)
                    .detach()
                    .cpu()
                )
            self.base_state_all = torch.stack(base_states, dim=0).to(self.device)

            points0 = self.points_at_time(self.xy, self.times[0])
            blocks = []
            for start in range(0, config.RANK, config.MODE_RANK_BLOCK):
                stop = min(start + config.MODE_RANK_BLOCK, config.RANK)
                directions = self.V[:, start:stop]
                block = vmap(
                    lambda direction: jvp(
                        lambda theta: kolmogorov_state_minimal_explicit(
                            self.model, theta, points0
                        ).reshape(-1),
                        (self.theta,),
                        (direction,),
                    )[1],
                    chunk_size=config.MODE_JVP_CHUNK,
                )(directions.T)
                blocks.append(
                    block.detach().reshape(stop - start, config.N_SPACE, 8).cpu()
                )
                print(
                    f"source={self.source_seed} cache modes " f"{start}:{stop}",
                    flush=True,
                )
            modes_cpu = torch.cat(blocks, dim=0)
            self.modes_state = modes_cpu.to(self.device)
            synchronize()
        self.cache_time = timer.seconds
        torch.save(
            {
                "xy": torch.tensor(self.xy),
                "times": torch.tensor(self.times),
                "base_state_all": self.base_state_all.detach().cpu(),
                "modes_state": modes_cpu,
                "rank": config.RANK,
                "features": [
                    "u",
                    "v",
                    "w",
                    "w_x",
                    "w_y",
                    "w_xx",
                    "w_yy",
                    "div",
                ],
                "coordinate_derivatives": "explicit_minimal_spatial",
                "cache_time_sec": self.cache_time,
            },
            self.mode_cache_path,
        )

    def prepare(self):
        self.extract_or_load_basis()
        self.build_or_load_cache()

    def nonlinear_term(self, state, points):
        force = -FORCE_N * torch.cos(FORCE_N * points[:, 1:2])
        return (
            state[:, 0:1] * state[:, 3:4]
            + state[:, 1:2] * state[:, 4:5]
            - (state[:, 5:6] + state[:, 6:7]) / config.RE
            - force
        )

    def step_residual(self, alpha, layer, previous_w, previous_nl):
        state = layer["base"] + torch.einsum("r,rnc->nc", alpha, self.modes_state)
        dt = float(layer["time"] - layer["previous_time"])
        nonlinear = self.nonlinear_term(state, layer["points"])
        equation = (state[:, 2:3] - previous_w) / dt + 0.5 * (nonlinear + previous_nl)
        return torch.cat([equation, state[:, 7:8]], dim=0).reshape(-1)

    def step_jacobian(self, alpha, layer):
        state = layer["base"] + torch.einsum("r,rnc->nc", alpha, self.modes_state)
        phi = self.modes_state.permute(1, 2, 0)
        dt = float(layer["time"] - layer["previous_time"])
        nonlinear = (
            phi[:, 0, :] * state[:, 3:4]
            + state[:, 0:1] * phi[:, 3, :]
            + phi[:, 1, :] * state[:, 4:5]
            + state[:, 1:2] * phi[:, 4, :]
            - (phi[:, 5, :] + phi[:, 6, :]) / config.RE
        )
        equation = phi[:, 2, :] / dt + 0.5 * nonlinear
        return torch.cat([equation, phi[:, 7, :]], dim=0)

    @staticmethod
    def solve_spd(matrix, rhs, diagonal_shift):
        matrix.diagonal().add_(diagonal_shift)
        factor = torch.linalg.cholesky(matrix)
        return torch.cholesky_solve(rhs.reshape(-1, 1), factor).reshape(-1)

    def solve(self, case, need_error=True):
        synchronize()
        online_start = time.perf_counter()
        target0 = torch.tensor(
            case["fields"][0].reshape(-1, 3),
            device=self.device,
            dtype=self.dtype,
        )
        initial_matrix = (
            self.modes_state[:, :, :3].permute(1, 2, 0).reshape(-1, config.RANK)
        )
        initial_rhs = (target0 - self.base_state_all[0, :, :3]).reshape(-1)
        alpha = self.solve_spd(
            initial_matrix.T @ initial_matrix,
            initial_matrix.T @ initial_rhs,
            config.LAMBDA_IC,
        )
        state = self.base_state_all[0] + torch.einsum(
            "r,rnc->nc", alpha, self.modes_state
        )
        initial_residual = (state[:, :3] - target0).reshape(-1)
        initial_projection_loss = float(initial_residual.square().mean().detach().cpu())
        previous_w = state[:, 2:3]
        points0 = self.points_at_time(self.xy, self.times[0])
        previous_nl = self.nonlinear_term(state, points0)
        previous_time = float(self.times[0])
        losses = []
        predictions = [state[:, :3].detach()] if need_error else None

        for index in range(1, config.NT):
            points = self.points_at_time(self.xy, self.times[index])
            layer = {
                "base": self.base_state_all[index],
                "points": points,
                "time": float(self.times[index]),
                "previous_time": previous_time,
            }
            residual = self.step_residual(alpha, layer, previous_w, previous_nl)
            jacobian = self.step_jacobian(alpha, layer)
            current_loss = float(residual.square().mean().detach().cpu())
            normal = jacobian.T @ jacobian
            rhs = -(jacobian.T @ residual)
            increment = self.solve_spd(normal, rhs, config.LM_INITIAL_MU)
            trial = alpha + increment
            trial_residual = self.step_residual(trial, layer, previous_w, previous_nl)
            trial_loss = float(trial_residual.square().mean().detach().cpu())
            if trial_loss < current_loss:
                alpha = trial
                loss = trial_loss
            else:
                loss = current_loss
            state = layer["base"] + torch.einsum("r,rnc->nc", alpha, self.modes_state)
            previous_w = state[:, 2:3]
            previous_nl = self.nonlinear_term(state, points)
            previous_time = float(self.times[index])
            losses.append(loss)
            if need_error:
                predictions.append(state[:, :3].detach())

        residual_loss = float(np.mean(losses))
        synchronize()
        solve_time = time.perf_counter() - online_start
        result = {
            "res_LST": residual_loss,
            "time_marching_loss": residual_loss,
            "initial_projection_loss": initial_projection_loss,
            "solve_time_sec": solve_time,
            "step_loss_mean": residual_loss,
            "step_loss_p90": float(np.quantile(losses, 0.90)),
            "step_loss_max": float(np.max(losses)),
        }

        if need_error:
            prediction = torch.stack(predictions, dim=0)
            truth = torch.tensor(
                case["fields"], device=self.device, dtype=self.dtype
            ).reshape(config.NT, config.N_SPACE, 3)
            global_error = torch.linalg.norm(
                (prediction - truth).reshape(-1)
            ) / torch.linalg.norm(truth.reshape(-1))
            slice_errors = torch.linalg.norm(
                (prediction - truth).reshape(config.NT, -1), dim=1
            ) / torch.linalg.norm(truth.reshape(config.NT, -1), dim=1)
            result.update(
                error_LST=float(global_error.detach().cpu()),
                mean_time_slice_error=float(slice_errors[1:].mean().detach().cpu()),
                peak_time_slice_error=float(slice_errors[1:].max().detach().cpu()),
                final_time_slice_error=float(slice_errors[-1].detach().cpu()),
                initial_projection_error=float(slice_errors[0].detach().cpu()),
            )
        return result
