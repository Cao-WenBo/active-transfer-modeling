import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state(seed):
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((config.GRF_MODES, config.GRF_MODES))
    k1, k2 = np.meshgrid(
        np.arange(config.GRF_MODES),
        np.arange(config.GRF_MODES),
        indexing="ij",
    )
    eig = config.TAU ** (config.ALPHA - 1) * (
        np.pi**2 * (k1 * k1 + k2 * k2) + config.TAU**2
    ) ** (-config.ALPHA / 2)
    norm = np.ones(config.GRF_MODES)
    norm[1:] = np.sqrt(2)
    amp = eig * norm[:, None] * norm[None, :]
    amp[0, 0] = 0
    return xi, amp


def _basis(x, modes):
    k = np.arange(modes, dtype=np.float64)
    phase = np.pi * np.asarray(x, dtype=np.float64).reshape(-1, 1) * k
    return np.cos(phase), -np.pi * k * np.sin(phase)


def eval_coeff_and_grad(case, points):
    points = np.asarray(points, dtype=np.float64)
    coefficients = case["grf_xi"] * case["grf_amp"]
    modes = coefficients.shape[0]
    cos_x, dcos_x = _basis(points[:, 0], modes)
    cos_y, dcos_y = _basis(points[:, 1], modes)
    field = np.einsum("nk,kl,nl->n", cos_x, coefficients, cos_y)
    field_x = np.einsum("nk,kl,nl->n", dcos_x, coefficients, cos_y)
    field_y = np.einsum("nk,kl,nl->n", cos_x, coefficients, dcos_y)
    permeability = np.exp(field)[:, None]
    return (
        permeability,
        permeability * field_x[:, None],
        permeability * field_y[:, None],
    )


def _uniform_cell_points(size):
    coordinate = (np.arange(size, dtype=np.float64) + 0.5) / size
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def _solve_reference(case, size):
    nodes = np.linspace(0.0, 1.0, size, dtype=np.float64)
    cells = (np.arange(size, dtype=np.float64) + 0.5) / size
    xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
    permeability = eval_coeff_and_grad(
        case, np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    )[0].reshape(size, size)
    forcing = RectBivariateSpline(cells, cells, np.ones((size, size)), kx=3, ky=3)(
        nodes, nodes
    )[1:-1, 1:-1]
    interior_size = size - 2
    matrix = lil_matrix(
        (interior_size * interior_size, interior_size * interior_size),
        dtype=np.float64,
    )

    def row(i, j):
        return j * interior_size + i

    for j in range(1, size - 1):
        for i in range(1, size - 1):
            index = row(i - 1, j - 1)
            east = 0.5 * (permeability[i, j] + permeability[i + 1, j])
            west = 0.5 * (permeability[i, j] + permeability[i - 1, j])
            north = 0.5 * (permeability[i, j] + permeability[i, j + 1])
            south = 0.5 * (permeability[i, j] + permeability[i, j - 1])
            matrix[index, index] = east + west + north + south
            if i < size - 2:
                matrix[index, row(i, j - 1)] = -east
            if i > 1:
                matrix[index, row(i - 2, j - 1)] = -west
            if j < size - 2:
                matrix[index, row(i - 1, j)] = -north
            if j > 1:
                matrix[index, row(i - 1, j - 2)] = -south

    rhs = forcing.T.reshape(-1) / (size - 1) ** 2
    interior = spsolve(matrix.tocsr(), rhs).reshape(interior_size, interior_size).T
    nodal_solution = np.zeros((size, size), dtype=np.float64)
    nodal_solution[1:-1, 1:-1] = interior
    return RectBivariateSpline(nodes, nodes, nodal_solution, kx=3, ky=3)(cells, cells)


def generate_case(seed):
    config.CASE_CACHE.mkdir(parents=True, exist_ok=True)
    path = config.CASE_CACHE / (
        f"case_seed{int(seed):05d}_field{config.FIELD_GRID_SIZE}_"
        f"reference{config.REFERENCE_GRID_SIZE}.npz"
    )
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    xi, amp = _state(seed)
    case = {"grf_xi": xi, "grf_amp": amp}
    field_points = _uniform_cell_points(config.FIELD_GRID_SIZE)
    reference_points = _uniform_cell_points(config.REFERENCE_GRID_SIZE)
    field_coefficients = eval_coeff_and_grad(case, field_points)[0]
    reference_solution = _solve_reference(case, config.REFERENCE_GRID_SIZE)
    case.update(
        seed=np.array(seed),
        field_grid_size=np.array(config.FIELD_GRID_SIZE),
        reference_grid_size=np.array(config.REFERENCE_GRID_SIZE),
        alpha=np.array(config.ALPHA),
        tau=np.array(config.TAU),
        modes=np.array(config.GRF_MODES),
        X_field=field_points,
        coeff_ref=field_coefficients,
        X_ref=reference_points,
        S_ref=reference_solution.reshape(-1, 1),
    )
    np.savez_compressed(path, **case)
    return case


def sample_residual(case, count, seed):
    points = np.random.default_rng(seed).random((count, 2))
    permeability, permeability_x, permeability_y = eval_coeff_and_grad(case, points)
    return {
        "X_r": points.astype(np.float32),
        "coeff_r": permeability.astype(np.float32),
        "coeff_x_r": permeability_x.astype(np.float32),
        "coeff_y_r": permeability_y.astype(np.float32),
    }
