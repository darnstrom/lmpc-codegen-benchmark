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

The `stm32_benchmark.py` script can build and measure solver performance entirely
automatically if you have an ARM embedded toolchain and an ST-LINK-equipped board.

### Quick start (automated)

```bash
# 1. Generate solver code (host)
uv run python run_all.py \
  --problem inverted_pendulum \
  --solvers lmpc casadi tinympc cvxpygen \
  --results-dir results

# 2. Build + flash + capture UART + summarize (one command)
uv run python stm32_benchmark.py run
```

`run` auto-detects the flash tool (st-flash / openocd / pyocd) and the serial
port.  After flashing each solver it waits for the BENCH_START…BENCH_END block
on UART2 (PA2/PA3, 115 200 Bd, 8N1) and saves a timing CSV.

Optional flags:
```
--solvers lmpc casadi        # limit to specific solvers
--problem double_integrator  # use a different problem
--port /dev/ttyACM0          # specify serial port explicitly
--flasher openocd            # choose flash tool
--timeout 120                # allow 120 s per solver
--no-flash                   # build only, skip flash/capture
```

### Step-by-step (manual)

If you prefer to flash manually or use a different IDE:

```bash
# 1. Build firmware for all solvers (requires arm-none-eabi-gcc + cmake)
uv run python stm32_benchmark.py build

# Binaries are in stm32_benchmark/build/<solver>/bench.bin

# 2. Flash each binary and capture UART output at 115 200 Bd.
#    The MCU prints one data line between BENCH_START and BENCH_END, e.g.:
#    BENCH_START
#    lmpc,5,50,96000000,125430,138002,121000,124800
#    BENCH_END
#    Fields: solver,n_warmup,n_timed,sysclk_hz,cycles_mean,max,min,median

# 3. Fill in stm32_benchmark/timings.csv and run the summarizer
uv run python stm32_benchmark.py summarize \
  --timings stm32_benchmark/timings.csv \
  --output-dir stm32_benchmark
```

### Prerequisites

| Requirement | Notes |
|---|---|
| `arm-none-eabi-gcc` + `cmake` | `sudo apt install cmake gcc-arm-none-eabi` |
| ST-LINK-equipped STM32F411 board | Nucleo-F411RE recommended |
| `st-flash` / `openocd` / `pyocd` | For automated flashing |
| `pip install pyserial` | For automated UART capture |

### Hardware interface (Nucleo-F411RE)

The firmware uses:
- **USART2** (PA2 TX / PA3 RX) — wired to the ST-LINK VCP on Nucleo boards
- **DWT cycle counter** — started once in `main()`, wraps at ~44 s @ 96 MHz
- **PLL**: 16 MHz HSI → 96 MHz SYSCLK, APB1 = 48 MHz

### Firmware template overview (`stm32/`)

| File | Purpose |
|---|---|
| `startup.c` | PLL init, .data/.bss init, C++ constructor call, vector table |
| `STM32F411CEUX_FLASH.ld` | Linker script (512 K Flash, 128 K RAM) |
| `arm_none_eabi.cmake` | CMake cross-toolchain file |
| `dwt_timer.h` | DWT cycle counter (header-only) |
| `uart.h` / `uart.c` | USART2 polling driver |
| `solver_adapter.h` | Solver-agnostic interface (`solver_init`, `solver_step`) |
| `bench_main.c` | Benchmark harness (warm-up + timed loop + UART output) |
| `adapters/lmpc_adapter.c` | Wraps `mpc_compute_control()` from lmpc codegen |
| `adapters/casadi_adapter.c` | Wraps CasADi qrqp `mpc_qp_cg()` |
| `adapters/tinympc_adapter.cpp` | Wraps TinyMPC C++ solver |
| `adapters/cvxpygen_adapter.c` | Wraps CVXPYgen/OSQP `cpg_solve()` |

Each adapter implements `solver_init()` and `solver_step(x0, r, u_prev, u_out)`.
The `build` command generates `problem_data.h` and `CMakeLists.txt` into
`stm32_benchmark/build/<solver>/` from the problem dimensions in
`problem_definition.py`, then drives the cross-compilation automatically.

### Solver-specific notes

- `lmpc`: plain C, easiest to integrate; `mpc_compute_control` manages `u_prev` internally.
- `casadi`: plain C (single self-contained `casadi_mpc.c`); qrqp solver is compact.
- `tinympc`: C++ with Eigen dependency (header-only, included in the generated tree); FPU strongly recommended.
- `cvxpygen`: embeds the full OSQP runtime as C sources in the generated tree.
- `acados`: depends on acados/BLASFEO/HPIPM runtime; not included in the automated build due to memory requirements.
