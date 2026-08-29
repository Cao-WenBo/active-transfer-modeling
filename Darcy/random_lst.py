"""Three-source single-source LST robustness experiment for Darcy."""

import json
import time
from pathlib import Path
import numpy as np
import torch
import config
from atm import build_solver, distance
from data_utils import generate_case, set_seed

SOURCE_SEEDS = [506, 969, 1008]
TARGET_SEEDS = list(config.TARGET_SEEDS)
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
    done = {(int(r["source_seed"]), int(r["target_seed"])) for r in rows}
    source_meta = {}
    for source_seed in SOURCE_SEEDS:
        t = time.perf_counter()
        solver, trained, train_time = build_solver(source_seed)
        source_meta[str(source_seed)] = {
            "trained_now": bool(trained),
            "source_error": solver.source_error,
            "source_loss": solver.source_loss,
            "recorded_train_time_sec": float(train_time),
            "build_wall_time_sec": time.perf_counter() - t,
        }
        for index, target_seed in enumerate(TARGET_SEEDS, 1):
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
                    f"source={source_seed} targets={index}/100 error={result['error_LST']:.6e}",
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
                "residual_points": config.TRANSFER_Q_TRAIN,
                "field_grid_size": config.FIELD_GRID_SIZE,
                "reference_grid_size": config.REFERENCE_GRID_SIZE,
                "error": "relative_L2_on_241x241_uniform_grid",
                "no_source_specific_tuning": True,
            },
            "summaries": summaries,
            "rows": rows,
        },
    )
    print(f"wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
