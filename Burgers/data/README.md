# Burgers dataset generation

The generated `Burger.mat` dataset is intentionally not distributed with this
repository. It can be recreated from the three MATLAB files in this directory:

- `GRF.m` samples periodic Gaussian random-field initial conditions;
- `Burgers.m` solves the viscous Burgers equation for one initial condition;
- `gen_Burgers.m` generates all cases and saves the assembled dataset.

## Requirements

- MATLAB;
- [Chebfun](https://www.chebfun.org/) installed and available on the MATLAB
  path. The solver uses Chebfun's `spinop` and `spin` functions.

## Generate the data

Start MATLAB, add Chebfun to the MATLAB path, and change to this `data`
directory. Then run:

```matlab
gen_Burgers
```

For a reproducible realization of the random initial conditions, set MATLAB's
random-number seed immediately before generation, for example:

```matlab
rng(0, 'twister');
gen_Burgers
```

Generation solves 1,200 trajectories on a 4,096-point internal spatial grid
with viscosity `0.01`, then samples each trajectory at 101 spatial locations
and 101 times on `[0, 1]`. Depending on the machine, this may take a substantial
amount of time.

The script writes `Burger.mat` in the current directory with these variables:

| Variable | Shape | Description |
|---|---:|---|
| `input` | `1200 x 101` | Sampled initial conditions |
| `output` | `1200 x 101 x 101` | Solution trajectories (time x space per case) |
| `tspan` | `1 x 101` | Time grid |
| `gamma`, `tau`, `sigma` | scalar | Gaussian random-field parameters |

Keep the output filename and location unchanged: the Python benchmark expects
`Burgers/data/Burger.mat`. After generation, return to the `Burgers` directory
and run `python atm.py` or `python random_lst.py`.
