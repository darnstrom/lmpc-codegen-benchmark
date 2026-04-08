# Benchmarking LinearMPC and related packages

This repository benchmarks the inverted-pendulum MPC problem across the solver wrappers in `benchmark.py`: `lmpc`, `casadi`, `cvxpygen`, `acados`, and `tinympc`.

## Python setup with uv

Install the Python dependencies from the lockfile with:

```bash
uv sync
```

The project is defined in `pyproject.toml`, and `uv.lock` captures the exact Python environment used for the benchmark scripts and report generation.

## System requirements

- A working Julia installation for `LinearMPC.jl` (will be installed together with lmpc)
- `git`, `cmake`, a C/C++ toolchain, and `make`
- For `acados`, a local acados build with shared libraries

One working acados setup is:

```bash
(
    git clone --recursive https://github.com/acados/acados.git /tmp/acados
    mkdir -p /tmp/acados/build
    cd /tmp/acados/build
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_INSTALL_DIR=/tmp/acados_install -DBUILD_SHARED_LIBS=ON ..
    make -j4 install
    mkdir -p /tmp/acados/bin
    export ACADOS_SOURCE_DIR=/tmp/acados
)
```


If acados is installed elsewhere, export `ACADOS_SOURCE_DIR` and either `ACADOS_INSTALL_DIR` or `ACADOS_LIB_PATH` before running the benchmark.

To install the Python interface, run 
```bash
uv run python -c "from acados_template import get_tera; print(get_tera(force_download=True))"
```

## Run the benchmark and generate plots

From the repository root:

```bash
uv run python run_all.py \
  --problem inverted_pendulum \
  --scaling \
  --solvers lmpc casadi cvxpygen acados tinympc \
  --horizons 50 75 100 125 \
  --results-dir results
uv run python generate_plots.py
```

## Output files

The command above writes:

- `results/benchmark_results.json`
- `results/benchmark_results_N50.json`
- `results/benchmark_results_N75.json`
- `results/benchmark_results_N100.json`
- `results/benchmark_results_N125.json`
- `results/scaling_results.json`
- `results/plots/scaling_time.png`
- `results/plots/scaling_memory.png`

Each per-horizon JSON contains setup time, solve-time statistics, code size, binary size, and the closed-loop trajectory for every solver.

To regenerate the plots from saved results without rerunning the benchmark:

```bash
uv run python generate_plot.py \
  --results-path results/benchmark_results.json \
  --plots-dir results/plots \
```

