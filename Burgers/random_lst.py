"""Three-source single-source LST robustness experiment for Burgers."""

import json
import time
from pathlib import Path
import numpy as np
import torch
import config
from atm import build_solver, distance, source_path
from data_utils import generate_case, set_seed

SOURCE_CASE_IDS = [537, 673, 1140]
TARGET_CASE_IDS = list(config.FORMAL_TARGET_CASE_IDS)
OUT = Path(__file__).resolve().parent / "results" / "random_lst"
CHECKPOINT = OUT / "checkpoint.json"
RESULT = OUT / "result.json"


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stats(rows, key):
    x = np.asarray([r[key] for r in rows], np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def main():
    set_seed(config.SEED)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    rows = (
        json.loads(CHECKPOINT.read_text(encoding="utf-8"))["rows"]
        if CHECKPOINT.exists()
        else []
    )
    done = {(int(r["source_case_id"]), int(r["target_case_id"])) for r in rows}
    source_meta = {}
    for source_case_id in SOURCE_CASE_IDS:
        existed = (source_path(source_case_id) / "solver_core.pt").exists()
        t = time.perf_counter()
        solver = build_solver(source_case_id)
        core = torch.load(
            source_path(source_case_id) / "solver_core.pt",
            map_location="cpu",
            weights_only=False,
        )
        source_meta[str(source_case_id)] = {
            "trained_now": not existed,
            "source_error": solver.source_error,
            "source_loss": solver.source_loss,
            "recorded_train_time_sec": float(
                core.get(
                    "train_time_sec",
                    core.get("log", {}).get("time", [0])[-1]
                    if core.get("log", {}).get("time")
                    else 0,
                )
            ),
            "build_wall_time_sec": time.perf_counter() - t,
        }
        for index, target_case_id in enumerate(TARGET_CASE_IDS, 1):
            if (source_case_id, target_case_id) in done:
                continue
            case = generate_case(target_case_id, config.M)
            result = solver.solve(case, need_error=True)
            result.update(
                source_case_id=source_case_id,
                target_case_id=target_case_id,
                distance=distance(solver.source_case, case),
            )
            rows.append(result)
            done.add((source_case_id, target_case_id))
            save(CHECKPOINT, {"rows": rows})
            if index % 10 == 0:
                print(
                    f"source={source_case_id} targets={index}/100 error={result['error_LST']:.6e}",
                    flush=True,
                )
    summaries = []
    for source_case_id in SOURCE_CASE_IDS:
        selected = [r for r in rows if int(r["source_case_id"]) == source_case_id]
        summaries.append(
            {
                "source_case_id": source_case_id,
                "n_targets": len(selected),
                "error": stats(selected, "error_LST"),
                "loss": stats(selected, "res_LST"),
                "online_time_sec": stats(selected, "solve_time_sec"),
                "distance": stats(selected, "distance"),
                "source": source_meta[str(source_case_id)],
            }
        )
    save(
        RESULT,
        {
            "protocol": {
                "source_case_ids": SOURCE_CASE_IDS,
                "target_case_ids": TARGET_CASE_IDS,
                "rank": config.RANK,
                "transfer_dtype": str(config.TRANSFER_DTYPE),
                "nonlinear_iters": config.NONLINEAR_ITERS,
                "no_source_specific_tuning": True,
            },
            "summaries": summaries,
            "rows": rows,
        },
    )
    print(f"wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
