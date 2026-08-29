import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import config
from data_utils import generate_case, set_seed
from source_solver import (
    expected_training_config as source_training_config,
    train_source,
)
from timing import Timer
from transfer_solver import TransferSolver


def source_path(seed):
    return config.SOURCE_ROOT / f"seed{int(seed):05d}"


def expected_training_config():
    return source_training_config()


def ensure_source(seed):
    path = source_path(seed)
    core_path = path / "solver_core.pt"
    if core_path.exists():
        core = torch.load(core_path, map_location="cpu", weights_only=False)
        if (
            core.get("derivatives") == "explicit_[u,u_x,u_y,u_xx,u_yy]"
            and core.get("training_config") == expected_training_config()
        ):
            return core_path, False
        raise RuntimeError(f"Incompatible source checkpoint: {core_path}")
    train_source(generate_case(seed), path)
    return core_path, True


def build_solver(seed):
    core_path, trained = ensure_source(seed)
    core = torch.load(core_path, map_location="cpu", weights_only=False)
    solver = TransferSolver(core)
    solver.prepare()
    set_seed(config.BASIS_SEED_BASE + int(seed))
    solver.extract_basis()
    solver.build_cache()
    return solver, trained, float(core["train_time_sec"])


def distance(a, b):
    delta = np.asarray(a["coeff_ref"], np.float64) - np.asarray(
        b["coeff_ref"], np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def evaluate_new_source(solver, source_seed, cases):
    rows = []
    for case in cases:
        result = solver.solve(case, need_error=False)
        result.update(
            seed=int(case["seed"]),
            selected_source_seed=int(source_seed),
            distance=distance(solver.source_case, case),
        )
        rows.append(result)
    return rows


def update_candidate_best(best, trial_rows):
    for row in trial_rows:
        seed = int(row["seed"])
        if seed not in best or float(row["res_LST"]) < float(best[seed]["res_LST"]):
            best[seed] = copy.deepcopy(row)


def evaluate_by_candidate_labels(solvers, candidates, candidate_best, cases):
    rows = []
    for case in cases:
        route_start = time.perf_counter()
        nearest = min(candidates, key=lambda candidate: distance(candidate, case))
        nearest_seed = int(nearest["seed"])
        selected = int(candidate_best[nearest_seed]["selected_source_seed"])
        route_time = time.perf_counter() - route_start
        result = solvers[selected].solve(case, need_error=True)
        result.update(
            seed=int(case["seed"]),
            selected_source_seed=selected,
            nearest_candidate_seed=nearest_seed,
            candidate_distance=distance(nearest, case),
            distance=distance(solvers[selected].source_case, case),
            route_time_sec=route_time,
            online_time_sec=route_time + float(result["solve_time_sec"]),
        )
        rows.append(result)
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


def candidate_seeds():
    return list(range(101, 201))


def save_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    set_seed(config.SEED)
    torch.set_num_threads(1)
    config.SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_ids = candidate_seeds()
    target_ids = config.TARGET_SEEDS
    candidates = [generate_case(seed) for seed in candidate_ids]
    targets = [generate_case(seed) for seed in target_ids]
    source_seeds = [config.INIT_SOURCE_SEED]
    registry, candidate_best, stages = {}, {}, []

    with Timer() as total_timer:
        for stage in range(1, config.MAX_SOURCES + 1):
            timing = {
                "source_train_actual_sec": 0.0,
                "source_train_recorded_sec": 0.0,
                "basis_extract_time_sec": 0.0,
                "derivative_cache_time_sec": 0.0,
            }
            new_source = int(source_seeds[-1])
            if new_source not in registry:
                solver, trained, train_time = build_solver(new_source)
                registry[new_source] = solver
                timing["source_train_actual_sec"] = train_time if trained else 0.0
                timing["source_train_recorded_sec"] = train_time
                timing["source_reused"] = not trained
                timing["basis_extract_time_sec"] = solver.basis_time
                timing["derivative_cache_time_sec"] = solver.cache_time

            with Timer() as candidate_timer:
                trial_rows = evaluate_new_source(
                    registry[new_source], new_source, candidates
                )
                update_candidate_best(candidate_best, trial_rows)
                candidate_rows = [
                    copy.deepcopy(candidate_best[int(case["seed"])])
                    for case in candidates
                ]
            with Timer() as evaluation_timer:
                target_rows = evaluate_by_candidate_labels(
                    registry, candidates, candidate_best, targets
                )

            timing.update(
                candidate_eval_time_sec=candidate_timer.seconds,
                external_test_eval_time_sec=evaluation_timer.seconds,
                candidate_transfer_solve_sum_sec=float(
                    sum(row["solve_time_sec"] for row in trial_rows)
                ),
                target_online_time_sum_sec=float(
                    sum(row["online_time_sec"] for row in target_rows)
                ),
            )
            candidate_loss = stats(candidate_rows, "res_LST")
            target_loss = stats(target_rows, "res_LST")
            target_error = stats(target_rows, "error_LST")
            online_stats = stats(target_rows, "online_time_sec")
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
                action, selected_stage, next_row = "rollback_and_stop", stage - 1, None
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
                "online_time_stats": online_stats,
                "delta_candidate_p90": delta,
                "stopping_action": action,
                "selected_stage_if_stop": selected_stage,
                "next_source_seed": None if next_row is None else int(next_row["seed"]),
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
                f"stage={stage} candidate_p90={candidate_loss['p90']:.6e} "
                f"delta={delta} mean_error={target_error['mean']:.6e} "
                f"action={action}",
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
            "transfer_dtype": "float64",
            "rank": config.RANK,
            "oversample": config.OVERSAMPLE,
            "basis_seed_base": config.BASIS_SEED_BASE,
            "lstsq_rcond": config.LSTSQ_RCOND,
            "source_Q_train": config.SOURCE_Q_TRAIN,
            "transfer_Q_train": config.TRANSFER_Q_TRAIN,
            "field_grid_size": config.FIELD_GRID_SIZE,
            "reference_grid_size": config.REFERENCE_GRID_SIZE,
            "candidate_routing": "minimum_cached_residual_over_active_sources",
            "deployment_routing": "top1_nearest_candidate_then_cached_source",
            "coordinate_derivatives": "explicit_minimal_[u,u_x,u_y,u_xx,u_yy]",
            "reference_evaluation": "chunked_parameter_JVP_on_241x241_grid",
            "stopping": "candidate_P90_10_percent_keep_nonworsening_current_stage",
            "candidate_seeds": candidate_ids,
            "target_seeds": target_ids,
        },
        "stages": stages,
        "selected_stage": selected_stage,
        "total_wall_time_sec": total_timer.seconds,
        "atm_offline_time_sec": float(
            sum(
                stage["timing"]["source_train_actual_sec"]
                + stage["timing"]["basis_extract_time_sec"]
                + stage["timing"]["derivative_cache_time_sec"]
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
