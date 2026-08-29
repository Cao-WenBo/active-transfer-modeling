import numpy as np
import torch

import config
from reference_data import generate_case as _gen
from reference_data import sample_training_data as _sample


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def generate_case(seed):
    return _gen(
        seed,
        config.LENGTH_SCALE,
        config.NX,
        config.NT,
        config.M,
    )


def sample_training_data(case):
    data = _sample(case, config.P_TRAIN, config.Q_TRAIN, 0)
    scale = float(case.get("velocity_scale", 1.0))
    if scale != 1.0:
        data = dict(data)
        data["ux_r"] = (scale * data["ux_r"]).astype(np.float32)
    return data
