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

## STM32F411 on-target benchmarking workflow

To compare solve time and firmware footprint on an STM32F411, first generate the solver code on the host:

```bash
uv run python run_all.py \
  --problem inverted_pendulum \
  --solvers lmpc casadi cvxpygen acados tinympc \
  --results-dir results
python stm32_benchmark.py prepare \
  --codegen-root codegen \
  --output-dir stm32_benchmark
```

The helper writes:

- `stm32_benchmark/manifest.json`: generated source files, include directories, language, and solver notes
- `stm32_benchmark/compile_codegen_objects.sh`: ARM cross-compile helper for object-only checks
- `stm32_benchmark/timings_template.csv`: template for the measurements collected on the board

### What you still need to do on the STM32 side

1. Install an ARM embedded toolchain (`arm-none-eabi-gcc`) and create an STM32F411 firmware project (CubeIDE, CMake, or Make-based is fine).
2. For each solver, import the source files and include directories listed in `manifest.json`.
3. Add a thin benchmark harness that:
   - initializes the generated solver,
   - runs a few warm-up solves,
   - measures repeated solves with the DWT cycle counter or another cycle-accurate timer,
   - prints one CSV row with `solver`, run counts, and timing statistics.
4. Build one firmware image per solver and save the size report, for example:
   ```bash
   mkdir -p stm32_benchmark/measurements
   arm-none-eabi-size build/lmpc.elf > stm32_benchmark/measurements/lmpc.size
   ```
5. Copy the measured timing values into `stm32_benchmark/timings_template.csv`.
6. Summarize the combined timing and size data:
   ```bash
python stm32_benchmark.py summarize \
     --manifest stm32_benchmark/manifest.json \
     --timings stm32_benchmark/timings_template.csv \
     --size-dir stm32_benchmark/measurements \
     --output-dir stm32_benchmark
   ```

This produces `summary.json`, `summary.csv`, and a console table with `solve_mean_us`, `solve_max_us`, `flash_bytes`, and `ram_bytes`.

### Solver-specific caveats

- `lmpc`: usually the easiest bare-metal target because the generated output is plain C.
- `casadi`: also plain C, but generated code size can be large.
- `cvxpygen`: import the whole generated subtree because the runtime support code lives inside it.
- `tinympc`: generated output is C++, so compile it with the C++ toolchain path in your firmware project.
- `acados`: depends on acados/BLASFEO/HPIPM runtime code; on STM32F411 it may need a custom reduced build or may exceed practical memory limits.
