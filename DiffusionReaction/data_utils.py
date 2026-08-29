import numpy as np
import torch
import config
from reference_data import generate_case as _generate_case
from reference_data import sample_training_data as _sample_training_data


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def generate_case(seed):
    return _generate_case(
        seed, config.LENGTH_SCALE, config.NX, config.NT, config.M, config.REFINE
    )


def sample_training_data(case, seed=0):
    return _sample_training_data(case, config.P_TRAIN, config.Q_TRAIN, seed)
