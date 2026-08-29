# Navier--Stokes

```bash
python atm.py
python random_lst.py
```

The Kolmogorov-flow initial vorticity is sampled from the Gaussian random field
specified in `config.py` and `data_utils.py`. Transfer uses strict float32
without TF32, rank 5000, and `(lambda_IC, mu, GN) = (1e-2, 10, 1)`. Each time
step uses the damped Cholesky solve defined in `transfer_solver.py`. Outputs are
written under `results/`.
