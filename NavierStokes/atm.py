import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import config
from data_utils import (
    GaussianRF2d,
    L,
    NavierStokes2d,
    generate_case,
    set_seed,
)
from source_solver import expected_training_config, train_source
from timing import Timer
from transfer_solver import TransferSolver


def source_path(seed):
    return config.SOURCE_ROOT / f"seed{int(seed):05d}"


def source_train_time(core):
    if "train_time_sec" in core:
        return float(core["train_time_sec"])
    log = core.get("log", {})
    times = log.get("time", [])
    return float(times[-1]) if times else 0.0


def ensure_source(seed):
    path = source_path(seed)
    core_path = path / "solver_core.pt"
    if core_path.exists():
        core = torch.load(core_path, map_location="cpu", weights_only=False)
        compatible = list(core.get("layers", [])) == config.LAYERS and (
            core.get("training_config") == expected_training_config()
            or (int(seed) == config.INITIAL_SOURCE_SEED)
        )
        if not compatible:
            raise RuntimeError(f"Incompatible source checkpoint: {core_path}")
        return core_path, False
    train_source(seed, path)
    return core_path, True


def build_solver(seed):
    core_path, trained = ensure_source(seed)
    core = torch.load(core_path, map_location="cpu", weights_only=False)
    solver = TransferSolver(core, seed)
    solver.prepare()
    return solver, trained, source_train_time(core)


def initial_condition(seed):
    solver = NavierStokes2d(config.NX, L, device="cpu", dtype=torch.float64)
    random_field = GaussianRF2d(config.NX, L, device="cpu", dtype=torch.float64)
    vorticity = random_field.sample(int(seed))
    u, v = solver.velocity_field(torch.fft.fft2(vorticity))
    return (
        torch.stack([u, v, vorticity], dim=-1).detach().cpu().numpy().astype(np.float32)
    )


def lightweight_candidate(seed):
    field = initial_condition(seed)
    return {"seed": int(seed), "fields": field[None]}


def condition(case):
    return np.asarray(case["fields"][0], dtype=np.float32)


def distance_fields(left, right):
    delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def evaluate_new_source(solver, source_seed, candidates):
    rows = []
    source_condition = initial_condition(source_seed)
    for index, case in enumerate(candidates):
        result = solver.solve(case, need_error=False)
        result.update(
            seed=int(case["seed"]),
            selected_source_seed=int(source_seed),
            distance=distance_fields(condition(case), source_condition),
        )
        rows.append(result)
        if (index + 1) % 10 == 0:
            print(
                f"source={source_seed} candidates={index + 1}/" f"{len(candidates)}",
                flush=True,
            )
    return rows


def update_candidate_best(best, trial_rows):
    for row in trial_rows:
        seed = int(row["seed"])
        if seed not in best or float(row["res_LST"]) < float(best[seed]["res_LST"]):
            best[seed] = copy.deepcopy(row)


def evaluate_targets(solvers, candidates, candidate_best, target_seeds):
    rows = []
    candidate_conditions = {int(case["seed"]): condition(case) for case in candidates}
    source_conditions = {int(seed): initial_condition(seed) for seed in solvers}
    for index, seed in enumerate(target_seeds):
        case = generate_case(
            seed=int(seed),
            Re=config.RE,
            nx=config.NX,
            nt=config.NT,
            T=config.TIME_INTERVAL,
            spinup=config.BURN_IN_TIME,
            device=config.DEVICE,
        )
        target_condition = condition(case)
        route_start = time.perf_counter()
        nearest_seed = min(
            candidate_conditions,
            key=lambda candidate_seed: distance_fields(
                candidate_conditions[candidate_seed], target_condition
            ),
        )
        selected = int(candidate_best[nearest_seed]["selected_source_seed"])
        route_time = time.perf_counter() - route_start
        result = solvers[selected].solve(case, need_error=True)
        result.update(
            seed=int(seed),
            selected_source_seed=selected,
            nearest_candidate_seed=int(nearest_seed),
            candidate_distance=distance_fields(
                candidate_conditions[nearest_seed], target_condition
            ),
            source_distance=distance_fields(
                source_conditions[selected], target_condition
            ),
            route_time_sec=route_time,
            online_time_sec=route_time + float(result["solve_time_sec"]),
        )
        rows.append(result)
        if (index + 1) % 10 == 0:
            print(
                f"targets={index + 1}/{len(target_seeds)}",
                flush=True,
            )
    return rows


def stats(rows, key):
    values = np.asarray([row[key] for row in rows], np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    set_seed(config.SEED)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config.SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    config.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_seeds = config.CANDIDATE_SEEDS
    target_seeds = config.TARGET_SEEDS
    candidates = [lightweight_candidate(seed) for seed in candidate_seeds]
    source_seeds = [config.INITIAL_SOURCE_SEED]
    registry, candidate_best, stages = {}, {}, []

    with Timer() as total_timer:
        for stage in range(1, config.MAX_SOURCES + 1):
            new_source = int(source_seeds[-1])
            solver, trained, train_time = build_solver(new_source)
            registry[new_source] = solver
            timing = {
                "source_train_actual_sec": train_time if trained else 0.0,
                "source_train_recorded_sec": train_time,
                "source_reused": not trained,
                "basis_extract_actual_sec": (
                    0.0 if solver.basis_reused else solver.basis_time
                ),
                "basis_extract_recorded_sec": solver.basis_time,
                "basis_reused": solver.basis_reused,
                "derivative_cache_actual_sec": (
                    0.0 if solver.cache_reused else solver.cache_time
                ),
                "derivative_cache_recorded_sec": solver.cache_time,
                "derivative_cache_reused": solver.cache_reused,
            }
            with Timer() as candidate_timer:
                trial_rows = evaluate_new_source(solver, new_source, candidates)
                update_candidate_best(candidate_best, trial_rows)
                candidate_rows = [
                    copy.deepcopy(candidate_best[int(case["seed"])])
                    for case in candidates
                ]
            with Timer() as evaluation_timer:
                target_rows = evaluate_targets(
                    registry,
                    candidates,
                    candidate_best,
                    target_seeds,
                )
            timing.update(
                candidate_eval_time_sec=candidate_timer.seconds,
                candidate_transfer_solve_sum_sec=float(
                    sum(row["solve_time_sec"] for row in trial_rows)
                ),
                external_test_eval_time_sec=evaluation_timer.seconds,
                target_online_time_sum_sec=float(
                    sum(row["online_time_sec"] for row in target_rows)
                ),
            )

            candidate_loss = stats(candidate_rows, "res_LST")
            target_loss = stats(target_rows, "res_LST")
            target_error = stats(target_rows, "error_LST")
            online_time = stats(target_rows, "online_time_sec")
            initial_loss = stats(target_rows, "initial_projection_loss")
            mean_slice_error = stats(target_rows, "mean_time_slice_error")
            previous = stages[-1]["candidate_loss_stats"]["p90"] if stages else None
            delta = (
                None
                if previous is None
                else (previous - candidate_loss["p90"]) / previous
            )
            remaining = [
                row for row in candidate_rows if int(row["seed"]) not in source_seeds
            ]
            next_row = (
                max(remaining, key=lambda row: row["res_LST"])
                if remaining and stage < config.MAX_SOURCES
                else None
            )
            action, selected_stage = "continue", stage
            if delta is not None and delta < 0:
                action, selected_stage, next_row = (
                    "rollback_and_stop",
                    stage - 1,
                    None,
                )
            elif delta is not None and delta < config.STOPPING_THRESHOLD:
                action, next_row = "keep_current_and_stop", None

            payload = {
                "stage": stage,
                "source_seeds": list(source_seeds),
                "candidate_results": candidate_rows,
                "target_results": target_rows,
                "candidate_loss_stats": candidate_loss,
                "target_loss_stats": target_loss,
                "error_stats": target_error,
                "mean_time_slice_error_stats": mean_slice_error,
                "initial_projection_loss_stats": initial_loss,
                "online_time_stats": online_time,
                "delta_candidate_p90": delta,
                "stopping_action": action,
                "selected_stage_if_stop": selected_stage,
                "next_source_seed": (
                    None if next_row is None else int(next_row["seed"])
                ),
                "timing": timing,
            }
            stages.append(payload)
            torch.save(payload, config.OUTPUT_ROOT / f"stage{stage:02d}.pt")
            save_json(
                config.OUTPUT_ROOT / f"stage{stage:02d}_summary.json",
                {
                    key: value
                    for key, value in payload.items()
                    if key not in ("candidate_results", "target_results")
                },
            )
            print(
                f"stage={stage} sources={source_seeds} "
                f"candidate_p90={candidate_loss['p90']:.6e} "
                f"delta={delta} global_error_mean="
                f"{target_error['mean']:.6e} action={action}",
                flush=True,
            )
            if action != "continue" or next_row is None:
                break
            source_seeds.append(int(next_row["seed"]))

    selected_stage = stages[-1]["selected_stage_if_stop"]
    result = {
        "config": {
            "network": config.LAYERS,
            "source_dtype": "float32",
            "transfer_dtype": "float32_strict_no_TF32",
            "rank": config.RANK,
            "oversample": config.OVERSAMPLE,
            "P_basis": config.P_BASIS,
            "basis_point_seed": config.BASIS_POINT_SEED,
            "condition_generation": {
                "grf_alpha": config.GRF_ALPHA,
                "grf_tau": config.GRF_TAU,
                "burn_in_time": config.BURN_IN_TIME,
            },
            "N_space": config.N_SPACE,
            "nt": config.NT,
            "gn_iters": config.GN_ITERS,
            "lambda_ic": config.LAMBDA_IC,
            "lambda_reg": 0.0,
            "lambda_diff": 0.0,
            "lm_initial_mu": config.LM_INITIAL_MU,
            "linear_solver": "damped_normal_cholesky",
            "response_basis": "[u,v,w]_on_10000_random_spacetime_points",
            "coordinate_derivatives": (
                "explicit_minimal_[u,v,w,w_x,w_y,w_xx,w_yy,div]"
            ),
            "candidate_routing": (
                "minimum_cached_time_marching_loss_over_active_sources"
            ),
            "deployment_routing": ("top1_nearest_candidate_then_cached_source"),
            "condition_distance": "initial_[u,v,w]_field_RMS",
            "stopping_loss": (
                "mean_of_64_equal_weight_vorticity_divergence_MSE;"
                "initial_projection_excluded"
            ),
            "stopping": ("candidate_P90_10_percent_keep_nonworsening_current_stage"),
            "primary_error": "global_spacetime_[u,v,w]_relative_L2",
            "candidate_seeds": candidate_seeds,
            "target_seeds": target_seeds,
        },
        "stages": stages,
        "selected_stage": selected_stage,
        "total_wall_time_sec": total_timer.seconds,
        "atm_offline_time_sec": float(
            sum(
                stage["timing"]["source_train_actual_sec"]
                + stage["timing"]["basis_extract_actual_sec"]
                + stage["timing"]["derivative_cache_actual_sec"]
                + stage["timing"]["candidate_eval_time_sec"]
                for stage in stages
            )
        ),
    }
    torch.save(result, config.OUTPUT_ROOT / "atm_results.pt")
    save_json(
        config.OUTPUT_ROOT / "final_summary.json",
        {
            "config": result["config"],
            "selected_stage": selected_stage,
            "total_wall_time_sec": result["total_wall_time_sec"],
            "atm_offline_time_sec": result["atm_offline_time_sec"],
            "stage_summaries": [
                {
                    key: value
                    for key, value in stage.items()
                    if key not in ("candidate_results", "target_results")
                }
                for stage in stages
            ],
        },
    )


if __name__ == "__main__":
    main()
