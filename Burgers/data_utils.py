from functools import lru_cache
import numpy as np
import scipy.io
import torch

from config import DATA_FILE


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@lru_cache(maxsize=1)
def load_dataset():
    data = scipy.io.loadmat(DATA_FILE)
    return (
        data["input"].astype(np.float32),
        data["output"].astype(np.float32),
        data["tspan"].reshape(-1).astype(np.float32),
    )


def generate_case(case_id, m=101):
    inputs, outputs, tspan = load_dataset()
    case_id = int(case_id)
    solution = outputs[case_id]
    u0 = solution[0]
    x = np.linspace(0.0, 1.0, solution.shape[1], dtype=np.float32)
    x_sensor = np.linspace(0.0, 1.0, m, dtype=np.float32)
    u_sensor = np.interp(x_sensor, x, u0).astype(np.float32)
    xx, tt = np.meshgrid(x, tspan, indexing="ij")
    return {
        "case_id": case_id,
        "m": int(m),
        "x": x,
        "t": tspan,
        "u0": u0.astype(np.float32),
        "u_sensor": u_sensor,
        "X_ref": np.column_stack([xx.reshape(-1), tt.reshape(-1)]).astype(np.float32),
        "S_ref": solution.T.reshape(-1, 1).astype(np.float32),
        "dataset_input": inputs[case_id].astype(np.float32),
    }


def sample_training_data(case, p_ic, p_bc, q_train, seed):
    rng = np.random.default_rng(seed)
    x_ic = np.linspace(0.0, 1.0, p_ic, dtype=np.float32)[:, None]
    s_ic = np.interp(x_ic[:, 0], case["x"], case["u0"])[:, None].astype(np.float32)
    t_bc = rng.random((p_bc, 1), dtype=np.float32)
    return {
        "X_ic": np.column_stack([x_ic[:, 0], np.zeros(p_ic, np.float32)]).astype(
            np.float32
        ),
        "s_ic": s_ic,
        "X_lbc": np.column_stack([np.zeros(p_bc, np.float32), t_bc[:, 0]]).astype(
            np.float32
        ),
        "X_ubc": np.column_stack([np.ones(p_bc, np.float32), t_bc[:, 0]]).astype(
            np.float32
        ),
        "X_r": rng.random((q_train, 2), dtype=np.float32),
    }
