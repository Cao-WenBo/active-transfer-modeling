from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "sources"
OUTPUT_ROOT = ROOT / "results" / "atm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [2, 20, 20, 20, 20, 1]
SOURCE_DTYPE = torch.float32
TRANSFER_DTYPE = torch.float64
LENGTH_SCALE = 0.2
NX = 100
NT = 100
M = 100
BASIS_SEED_BASE = 100000
SEED = 0
INIT_SOURCE_SEED = 0
TARGET_SEEDS = list(range(1, 101))
NUM_CANDIDATES = 100
MAX_SOURCES = 5
P_TRAIN = 1000
Q_TRAIN = 10000
SOURCE_OUTER_STEPS = 30
LBFGS_MAX_ITER = 300
RANK = 1000
OVERSAMPLE = 20
CHUNK_SIZE = 16
LSTSQ_RCOND = 1e-15
STOPPING_THRESHOLD = 0.1
