import copy
import json
from pathlib import Path
import time
import numpy as np
import torch

import config
from data_utils import generate_case, load_dataset, set_seed
from source_solver import train_source
from timing import Timer
from transfer_solver import TransferSolver


def source_name(case_id):
    return f"case{int(case_id):04d}"


def source_path(case_id):
    return config.SOURCE_ROOT / source_name(case_id)


def ensure_source(case_id):
    path = source_path(case_id)
    core = path / "solver_core.pt"
    if core.exists():
        payload = torch.load(core, map_location="cpu", weights_only=False)
        expected = {
            "P_IC": config.P_IC,
            "P_BC": config.P_BC,
            "Q_TRAIN": config.Q_TRAIN,
            "SOURCE_MIN_ITERATIONS": config.SOURCE_MIN_ITERATIONS,
            "LBFGS_OUTER_STEPS": config.LBFGS_OUTER_STEPS,
            "LBFGS_MAX_ITER": config.LBFGS_MAX_ITER,
            "LBFGS_RESAMPLE": config.LBFGS_RESAMPLE,
            "seed": 0,
        }
        compatible = (
            payload.get("derivatives") == "explicit_minimal_[u,ux,ut,uxx]"
            and payload.get("training_config") == expected
        )
        if compatible:
            return core
        raise RuntimeError(f"Non-formal or incompatible source checkpoint: {core}")
    train_source(generate_case(case_id, config.M), path)
    return core


def build_solver(case_id):
    core_path = ensure_source(case_id)
    core = torch.load(core_path, map_location="cpu", weights_only=False)
    solver = TransferSolver(core)
    solver.prepare_points(solver.source_case, seed=0)
    set_seed(config.BASIS_SEED_BASE + int(case_id))
    solver.extract_basis()
    solver.build_cache()
    return solver


def distance(source_case, target_case):
    d = np.asarray(source_case["u_sensor"], np.float64) - np.asarray(
        target_case["u_sensor"], np.float64
    )
    return float(np.sqrt(np.mean(d * d)))


def choose_nearest(solvers, case):
    ranked = sorted(
        (distance(s.source_case, case), case_id) for case_id, s in solvers.items()
    )
    return ranked[0][1], ranked[0][0]


def evaluate(solvers, cases, need_error):
    rows = []
    for case in cases:
        selected, dist = choose_nearest(solvers, case)
        result = solvers[selected].solve(case, need_error=need_error)
        result.update(
            {
                "case_id": int(case["case_id"]),
                "selected_source_case_id": int(selected),
                "distance": dist,
            }
        )
        rows.append(result)
    return rows


def evaluate_new_source(solver, source_id, cases):
    """Evaluate only the newly added source on every fixed candidate."""
    rows = []
    for case in cases:
        result = solver.solve(case, need_error=False)
        result.update(
            {
                "case_id": int(case["case_id"]),
                "selected_source_case_id": int(source_id),
                "distance": distance(solver.source_case, case),
            }
        )
        rows.append(result)
    return rows


def update_candidate_best(best, trial_rows):
    for row in trial_rows:
        case_id = int(row["case_id"])
        if case_id not in best or float(row["res_LST"]) < float(
            best[case_id]["res_LST"]
        ):
            best[case_id] = copy.deepcopy(row)


def evaluate_by_candidate_labels(
    solvers, candidates, candidate_best, cases, need_error
):
    """Deployment route: nearest candidate -> its cached minimum-loss source."""
    rows = []
    for case in cases:
        route_start = time.perf_counter()
        nearest = min(candidates, key=lambda candidate: distance(candidate, case))
        nearest_id = int(nearest["case_id"])
        selected = int(candidate_best[nearest_id]["selected_source_case_id"])
        route_time = time.perf_counter() - route_start
        result = solvers[selected].solve(case, need_error=need_error)
        result.update(
            {
                "case_id": int(case["case_id"]),
                "selected_source_case_id": selected,
                "nearest_candidate_case_id": nearest_id,
                "candidate_distance": distance(nearest, case),
                "distance": distance(solvers[selected].source_case, case),
                "route_time_sec": route_time,
                "online_time_sec": route_time + float(result["solve_time_sec"]),
            }
        )
        rows.append(result)
    return rows


def stats(rows, key):
    a = np.asarray([r[key] for r in rows], np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def candidate_ids():
    n_cases = load_dataset()[0].shape[0]
    excluded = {config.INIT_SOURCE_CASE_ID, *config.FORMAL_TARGET_CASE_IDS}
    available = np.asarray([i for i in range(n_cases) if i not in excluded], np.int64)
    rng = np.random.default_rng(config.CANDIDATE_POOL_SEED)
    return [int(i) for i in rng.choice(available, config.NUM_CANDIDATES, replace=False)]


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    set_seed(config.SEED)
    torch.set_num_threads(1)
    config.SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cids = candidate_ids()
    tids = config.FORMAL_TARGET_CASE_IDS
    candidates = [generate_case(i, config.M) for i in cids]
    targets = [generate_case(i, config.M) for i in tids]
    source_ids, stages = [config.INIT_SOURCE_CASE_ID], []
    solver_registry, candidate_best = {}, {}

    with Timer() as total_timer:
        for stage in range(1, config.MAX_SOURCES + 1):
            timing = {
                "source_train_time_sec": 0.0,
                "basis_extract_time_sec": 0.0,
                "derivative_cache_time_sec": 0.0,
            }
            for sid in source_ids:
                if sid not in solver_registry:
                    existed = (source_path(sid) / "solver_core.pt").exists()
                    with Timer() as build_timer:
                        solver_registry[sid] = build_solver(sid)
                    solver = solver_registry[sid]
                    timing["source_train_time_sec"] += (
                        0.0
                        if existed
                        else max(
                            0.0,
                            build_timer.seconds - solver.basis_time - solver.cache_time,
                        )
                    )
                    timing["basis_extract_time_sec"] += solver.basis_time
                    timing["derivative_cache_time_sec"] += solver.cache_time
            with Timer() as candidate_timer:
                new_source_id = int(source_ids[-1])
                trial_rows = evaluate_new_source(
                    solver_registry[new_source_id], new_source_id, candidates
                )
                update_candidate_best(candidate_best, trial_rows)
                candidate_rows = [
                    copy.deepcopy(candidate_best[int(c["case_id"])]) for c in candidates
                ]
            with Timer() as target_timer:
                target_rows = evaluate_by_candidate_labels(
                    solver_registry,
                    candidates,
                    candidate_best,
                    targets,
                    need_error=True,
                )
            timing["candidate_eval_time_sec"] = candidate_timer.seconds
            timing["external_test_eval_time_sec"] = target_timer.seconds
            timing["candidate_transfer_solve_sum_sec"] = float(
                sum(r["solve_time_sec"] for r in trial_rows)
            )
            timing["target_online_time_sum_sec"] = float(
                sum(r["online_time_sec"] for r in target_rows)
            )

            remaining = [r for r in candidate_rows if r["case_id"] not in source_ids]
            next_case = (
                None
                if stage == config.MAX_SOURCES
                else max(remaining, key=lambda r: r["res_LST"])
            )
            candidate_loss_stats = stats(candidate_rows, "res_LST")
            target_loss_stats = stats(target_rows, "res_LST")
            error_stats = stats(target_rows, "error_LST")
            online_time_stats = stats(target_rows, "online_time_sec")
            previous_p90 = stages[-1]["candidate_loss_stats"]["p90"] if stages else None
            delta_p90 = (
                None
                if previous_p90 is None
                else (previous_p90 - candidate_loss_stats["p90"]) / previous_p90
            )
            stop_action = "continue"
            selected_stage = stage
            if delta_p90 is not None and delta_p90 < 0:
                stop_action, selected_stage = "rollback_and_stop", stage - 1
            elif delta_p90 is not None and delta_p90 < config.STOPPING_THRESHOLD:
                stop_action, selected_stage = "keep_current_and_stop", stage

            stage_payload = {
                "stage": stage,
                "source_case_ids": list(source_ids),
                "candidate_results": candidate_rows,
                "target_results": target_rows,
                "candidate_loss_stats": candidate_loss_stats,
                "target_loss_stats": target_loss_stats,
                "loss_stats": target_loss_stats,
                "error_stats": error_stats,
                "online_time_stats": online_time_stats,
                "delta_candidate_p90": delta_p90,
                "delta_p90": delta_p90,
                "stopping_action": stop_action,
                "selected_stage_if_stop": selected_stage,
                "next_source_case_id": None
                if next_case is None
                else int(next_case["case_id"]),
                "timing": timing,
            }
            stages.append(stage_payload)
            torch.save(stage_payload, config.OUTPUT_ROOT / f"stage{stage:02d}.pt")
            save_json(
                config.OUTPUT_ROOT / f"stage{stage:02d}_summary.json",
                {
                    k: v
                    for k, v in stage_payload.items()
                    if k not in ("candidate_results", "target_results")
                },
            )
            if stop_action != "continue" or next_case is None:
                break
            source_ids.append(int(next_case["case_id"]))

    final_stage = stages[-1]["selected_stage_if_stop"]
    payload = {
        "config": {
            "network": config.LAYERS,
            "source_dtype": "float32",
            "transfer_dtype": "float64",
            "rank": config.RANK,
            "oversample": config.OVERSAMPLE,
            "basis_seed_base": config.BASIS_SEED_BASE,
            "lstsq_rcond": config.LSTSQ_RCOND,
            "P_ic": config.P_IC,
            "P_bc": config.P_BC,
            "Q_train": config.Q_TRAIN,
            "candidate_routing": "minimum_cached_residual_over_active_sources",
            "deployment_routing": "top1_nearest_candidate_then_cached_source",
            "routing": "loss_guided_candidate_top1_deployment",
            "coordinate_derivatives": "explicit_minimal_[u,ux,ut,uxx]",
            "parameter_derivatives": "jvp_vjp",
            "reference_cache": "uniform_grid_response_values_only",
            "stopping": "candidate_P90_10_percent_keep_nonworsening_current_stage",
            "candidate_case_ids": cids,
            "target_case_ids": tids,
        },
        "stages": stages,
        "selected_stage": final_stage,
        "total_wall_time_sec": total_timer.seconds,
        "atm_offline_time_sec": float(
            sum(
                stage["timing"]["source_train_time_sec"]
                + stage["timing"]["basis_extract_time_sec"]
                + stage["timing"]["derivative_cache_time_sec"]
                + stage["timing"]["candidate_eval_time_sec"]
                for stage in stages
            )
        ),
    }
    torch.save(payload, config.OUTPUT_ROOT / "atm_results.pt")
    save_json(
        config.OUTPUT_ROOT / "final_summary.json",
        {
            "config": payload["config"],
            "selected_stage": final_stage,
            "total_wall_time_sec": total_timer.seconds,
            "atm_offline_time_sec": payload["atm_offline_time_sec"],
            "stage_summaries": [
                {
                    k: v
                    for k, v in s.items()
                    if k not in ("candidate_results", "target_results")
                }
                for s in stages
            ],
        },
    )
    print(f"selected_stage={final_stage} total_wall_time={total_timer.seconds:.3f}s")


if __name__ == "__main__":
    main()
