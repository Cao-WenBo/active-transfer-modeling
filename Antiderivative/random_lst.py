"""Three-source single-source LST robustness experiment for Antiderivative."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import config
from atm import build_solver, distance
from data_utils import generate_case, set_seed


SOURCE_SEEDS = [458, 625, 789]
TARGET_SEEDS = list(config.TARGET_SEEDS)
OUT = Path(__file__).resolve().parent / "results" / "random_lst"
CHECKPOINT = OUT / "checkpoint.json"
RESULT = OUT / "result.json"


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stats(rows, key):
    values = np.asarray([row[key] for row in rows], np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main():
    set_seed(config.SEED)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if CHECKPOINT.exists():
        rows = json.loads(CHECKPOINT.read_text(encoding="utf-8"))["rows"]
    done = {(int(r["source_seed"]), int(r["target_seed"])) for r in rows}
    source_meta = {}

    for source_seed in SOURCE_SEEDS:
        build_start = time.perf_counter()
        solver, trained, train_time = build_solver(source_seed)
        source_meta[str(source_seed)] = {
            "trained_now": bool(trained),
            "source_error": solver.source_error,
            "source_loss": solver.source_loss,
            "recorded_train_time_sec": float(train_time),
            "build_wall_time_sec": time.perf_counter() - build_start,
        }
        for index, target_seed in enumerate(TARGET_SEEDS, start=1):
            if (source_seed, target_seed) in done:
                continue
            case = generate_case(target_seed)
            result = solver.solve(case, need_error=True)
            result.update(
                source_seed=source_seed,
                target_seed=target_seed,
                distance=distance(solver.source_case, case),
            )
            rows.append(result)
            done.add((source_seed, target_seed))
            save(CHECKPOINT, {"rows": rows})
            if index % 10 == 0:
                print(
                    f"source={source_seed} targets={index}/100 "
                    f"error={result['error_LST']:.6e}",
                    flush=True,
                )

    summaries = []
    for source_seed in SOURCE_SEEDS:
        selected = [r for r in rows if int(r["source_seed"]) == source_seed]
        summaries.append(
            {
                "source_seed": source_seed,
                "n_targets": len(selected),
                "error": stats(selected, "error_LST"),
                "loss": stats(selected, "res_LST"),
                "online_time_sec": stats(selected, "solve_time_sec"),
                "distance": stats(selected, "distance"),
                "source": source_meta[str(source_seed)],
            }
        )
    save(
        RESULT,
        {
            "protocol": {
                "source_seeds": SOURCE_SEEDS,
                "target_seeds": TARGET_SEEDS,
                "rank": config.RANK,
                "transfer_dtype": str(config.TRANSFER_DTYPE),
                "no_source_specific_tuning": True,
            },
            "summaries": summaries,
            "rows": rows,
        },
    )
    print(f"wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
