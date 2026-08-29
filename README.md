# Single-condition neural solvers encode transferable response spaces for parametric differential equations

Official implementation of **Linearized Subspace Transfer (LST)** and
**Active Transfer Modeling (ATM)**.

## Paper

**Authors:** Wenbo Cao<sup>1,2</sup> and Weiwei Zhang<sup>3,4,5,*</sup>

1. Institute of AI for Industries, Chinese Academy of Sciences, Nanjing
   211135, China
2. Institute of Computing Technology, Chinese Academy of Sciences, Beijing
   100190, China
3. School of Aeronautics, Northwestern Polytechnical University, Xi'an
   710072, China
4. International Joint Institute of Artificial Intelligence on Fluid
   Mechanics, Northwestern Polytechnical University, Xi'an 710072, China
5. National Key Laboratory of Aircraft Configuration Design, Xi'an 710072,
   China

<sup>*</sup>Corresponding author: aeroelastic@nwpu.edu.cn

### Abstract

Operator learning for parametric partial differential equations (PDEs)
typically builds global models over prescribed domains, requiring
cross-condition data or costly physics-constrained training. Here we show that
the output Jacobian of a neural solution model trained at one condition defines
a reusable response space for cross-condition solution variations. We
introduce Linearized Subspace Transfer (LST) to exploit this space and recover
target solutions by minimizing the target PDE-system residual over
response-space coordinates. Because any single response space has finite
coverage, Active Transfer Modeling (ATM) uses post-transfer residuals as
coverage indicators to selectively acquire response spaces from additional
single-condition models. Across six systems, single-condition response spaces
supported cross-condition transfer, with enrichment improving accuracy when
added spaces expanded representation capacity. Relative to evaluated
physics-informed operator baselines, ATM reduced error and offline construction
cost, with orders-of-magnitude accuracy gains in representative cases and
millisecond-to-second target adaptation. These results establish neural solvers
as reusable local parametric models.

## Repository overview

This package contains the six benchmark implementations used for the active
transfer modeling experiments. Each benchmark directory is self-contained and
provides two entry points:

- `atm.py`: multi-source ATM construction;
- `random_lst.py`: single-source LST with three random source conditions and
  100 held-out target conditions.

## Benchmark configurations

| Benchmark | Source network | Transfer precision | Rank | Coordinate updates |
|---|---:|---:|---:|---:|
| Antiderivative | 20 x 4 tanh MLP | float64 | 40 | 1 direct LS |
| Diffusion--Reaction | 20 x 4 tanh MLP | float64 | 1000 | 3 GN updates |
| Burgers | 20 x 4 tanh MLP | float64 | 1000 | 5 GN updates |
| Advection | 20 x 4 tanh MLP | float64 | 1000 | 1 direct LS |
| Darcy | 32 x 6 tanh MLP | float64 | 2500 | 1 direct LS |
| Navier--Stokes | Fourier features + 32 x 6 tanh MLP | float32 | 5000 | 1 damped GN update per time step |

All source PINNs and response-coordinate caches use explicit coordinate
derivatives containing only the channels required by the governing equation.
The response-space oversampling parameter is 20 for every benchmark.

The initial source, candidate pool, and held-out target set are disjoint. ATM
stores the lowest transfer loss and source label for every candidate. A target
is routed to its nearest candidate and uses that candidate's stored source
label. The stopping statistic is the candidate-pool P90 loss. The current stage
is retained when its relative reduction is nonnegative and below 10%; a stage
with increased P90 loss is rejected.

## Environment

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m pip install -r requirements.txt
```

The reference environment used Python 3.11, PyTorch 2.10, NumPy 2.3, and SciPy
1.17. CUDA and PyTorch builds are not pinned to a single platform.

## Running the experiments

Run each benchmark independently from its own directory.

```bash
cd Burgers
python atm.py
python random_lst.py
```

The ATM summary is written to `results/atm/final_summary.json`. The random-source
LST result is written to `results/random_lst/result.json`. Source checkpoints,
reference cases, and response caches are generated inside the benchmark
directory and reused on subsequent runs.

## Reference results

| Benchmark | Retained stage | Final mean relative L2 error |
|---|---:|---:|
| Antiderivative | 2 | 9.726e-8 |
| Diffusion--Reaction | 3 | 8.217e-5 |
| Burgers | 5 | 1.195e-5 |
| Advection | 3 | 4.568e-3 |
| Darcy | 3 | 2.136e-4 |
| Navier--Stokes | 2 | 4.521e-2 |

Machine-readable ATM and random-source LST reference results are stored under
each benchmark's `results/` directory.

## License

This project is released under the [MIT License](LICENSE).
