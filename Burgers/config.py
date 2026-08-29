from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "Burger.mat"
SOURCE_ROOT = ROOT / "sources"
OUTPUT_ROOT = ROOT / "results" / "atm"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [2, 20, 20, 20, 20, 1]
TRAIN_DTYPE = torch.float32
TRANSFER_DTYPE = torch.float64

SEED = 0
BASIS_SEED_BASE = 100000
INIT_SOURCE_CASE_ID = 0
MAX_SOURCES = 5
NUM_CANDIDATES = 100
CANDIDATE_POOL_SEED = 0
FORMAL_TARGET_CASE_IDS = list(range(1, 101))

M = 101
P_IC = 101
P_BC = 1000
Q_TRAIN = 10000
RANK = 1000
OVERSAMPLE = 20
CHUNK_SIZE = 16
NONLINEAR_ITERS = 5
LSTSQ_RCOND = 1e-15

SOURCE_MIN_ITERATIONS = 10000
LBFGS_OUTER_STEPS = 80
LBFGS_MAX_ITER = 300
LBFGS_RESAMPLE = False
NU = 0.01

# Candidate-pool P90 improvement controls whether another source is explored.
STOPPING_THRESHOLD = 0.10
