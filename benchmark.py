"""
Closed-loop benchmark for all MPC solvers.

Runs each solver in a closed-loop simulation, measures per-step solve time,
and collects results for reporting.
"""

import time
import json
import os
import sys
import traceback
import subprocess

import numpy as np

# Add project root to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from problem_definition import get_problem


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_closed_loop(solver, prob: dict, n_warmup: int = 5) -> dict:
    """
    Run a closed-loop MPC simulation.

    Discards the first `n_warmup` solve times (JIT / caching effects) before
    recording statistics.

    Supports a time-varying reference trajectory via ``prob['r_traj']``
    (shape ``(n_steps, ny)``).  Falls back to the constant ``prob['r_sim']``
    if ``r_traj`` is absent.

    Returns a result dict with keys:
        states, controls, solve_times_s, t_vec
    """
    Ad    = prob["Ad"]
    Bd    = prob["Bd"]
    x0    = prob["x0_sim"].copy()
    r     = prob["r_sim"].copy()
    r_traj = prob.get("r_traj", None)  # optional (n_steps, ny) array
    n     = prob["n_steps"]
    Ts    = prob["Ts"]
    nu    = prob["nu"]

    x        = x0.copy()
    u_prev   = np.zeros(nu)
    states   = [x.copy()]
    controls = []
    refs     = []
    times    = []

    for k in range(n):
        r_k = r_traj[k] if r_traj is not None else r

        reps = int(getattr(solver, "timing_repetitions", 1))
        t0 = time.perf_counter()
        u = None
        for _ in range(reps):
            u = solver.solve(x, r_k, u_prev)
        t1 = time.perf_counter()

        if k >= n_warmup:
            times.append((t1 - t0) / reps)

        u      = np.atleast_1d(u).flatten()
        controls.append(u.copy())
        refs.append(r_k.copy())

        # Euler step with zero-order hold (use discrete model)
        x      = Ad @ x + (Bd @ u).flatten()  # works for any nu
        states.append(x.copy())
        u_prev = u.copy()

    return {
        "states":        np.array(states),         # (n+1, nx)
        "controls":      np.array(controls),        # (n, nu)
        "references":    np.array(refs),            # (n, ny) — for plotting
        "solve_times_s": np.array(times),           # (n - n_warmup,)
        "t_vec":         np.arange(n + 1) * Ts,    # (n+1,)
    }


# ---------------------------------------------------------------------------
# Memory / code footprint helpers
# ---------------------------------------------------------------------------

def measure_code_footprint(directory: str) -> dict:
    """Count lines and bytes of C source files in a directory tree."""
    total_lines = 0
    total_bytes = 0
    file_count  = 0

    if not os.path.isdir(directory):
        return {"code_lines": 0, "code_bytes": 0, "c_files": 0}

    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith((".c", ".h", ".cpp", ".hpp")):
                path = os.path.join(root, fname)
                size = os.path.getsize(path)
                total_bytes += size
                file_count  += 1
                with open(path, "r", errors="replace") as f:
                    total_lines += sum(1 for _ in f)

    return {
        "code_lines": total_lines,
        "code_bytes": total_bytes,
        "c_files":    file_count,
    }


def measure_binary_footprint(directory: str) -> int:
    """Return total bytes of compiled binaries (.so, .a, .o) in directory."""
    total = 0
    if not os.path.isdir(directory):
        return 0
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith((".so", ".a", ".o", ".dylib")):
                total += os.path.getsize(os.path.join(root, fname))
    return total


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

SOLVER_REGISTRY = {
    "lmpc":      ("solvers.lmpc_solver",      "LMPCSolver"),
    "casadi":    ("solvers.casadi_solver",     "CasadiSolver"),
    "cvxpygen":  ("solvers.cvxpygen_solver",   "CvxpygenSolver"),
    "acados":    ("solvers.acados_solver",     "AcadosSolver"),
    "tinympc":   ("solvers.tinympc_solver",    "TinyMPCSolver"),
}

CODEGEN_BASE = os.path.join(_ROOT, "codegen")


def load_solver(solver_name: str, prob: dict):
    """Dynamically import and instantiate a solver."""
    module_path, class_name = SOLVER_REGISTRY[solver_name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(prob)


def run_benchmark(
    problem_name: str = "inverted_pendulum",
    solvers: list = None,
    results_dir: str = None,
    skip_codegen: bool = False,
    problem_overrides: dict | None = None,
    output_name: str = "benchmark_results.json",
) -> dict:
    """
    Run the full benchmark for all (or selected) solvers.

    Args:
        problem_name:  Problem from problem_definition.get_problem()
        solvers:       List of solver names to run (default: all registered)
        results_dir:   Where to save results JSON
        skip_codegen:  If True, skip code generation (use existing codegen dirs)

    Returns:
        results dict (also saved to JSON)
    """
    if solvers is None:
        solvers = list(SOLVER_REGISTRY.keys())

    if results_dir is None:
        results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    prob = get_problem(problem_name, overrides=problem_overrides)
    print(f"\n{'='*60}")
    print(f"  Benchmark: {prob['name']}")
    print(f"  nx={prob['nx']} nu={prob['nu']} ny={prob['ny']}  "
          f"Np={prob['Np']} Nc={prob['Nc']}  Ts={prob['Ts']}s  "
          f"sim_steps={prob['n_steps']}")
    print(f"{'='*60}\n")

    all_results = {
        "problem": problem_name,
        "problem_overrides": problem_overrides or {},
        "solvers": {},
    }

    for sname in solvers:
        print(f"--- {sname.upper()} ---")
        result = {}

        try:
            solver = load_solver(sname, prob)

            codegen_dir = os.path.join(CODEGEN_BASE, f"{sname}_codegen")

            # --- Setup / code generation ---
            if not skip_codegen:
                print(f"  Setting up ({sname})...", flush=True)
                t_setup_start = time.perf_counter()
                setup_info    = solver.setup(codegen_dir=codegen_dir)
                t_setup_end   = time.perf_counter()
                result["setup_time_s"] = t_setup_end - t_setup_start
                print(f"  Setup done in {result['setup_time_s']:.1f}s")
            else:
                # Re-init without regenerating code
                setup_info = solver.setup(codegen_dir=codegen_dir)

            # --- Closed-loop simulation ---
            print(f"  Running closed-loop simulation...", flush=True)
            sim = simulate_closed_loop(solver, prob)

            times = sim["solve_times_s"]
            result.update({
                "solve_time_mean_ms":   float(np.mean(times) * 1e3),
                "solve_time_median_ms": float(np.median(times) * 1e3),
                "solve_time_std_ms":    float(np.std(times) * 1e3),
                "solve_time_max_ms":    float(np.max(times) * 1e3),
                "solve_time_min_ms":    float(np.min(times) * 1e3),
                "solve_times_ms":       (times * 1e3).tolist(),  # for box plot
            })

            # --- Code / memory footprint ---
            footprint = measure_code_footprint(codegen_dir)
            result.update(footprint)

            metrics = solver.get_metrics()
            result.update(metrics)

            # Use library_bytes from solver (actual compiled binary: .a or .so).
            # Fall back to measuring the directory if not provided.
            lib_b = result.get("library_bytes") or 0
            result["binary_bytes"] = lib_b
            # acados also reports runtime deps (BLASFEO+HPIPM+acados shared libs)
            result.setdefault("runtime_deps_bytes", 0)

            # --- Trajectories (save for plotting) ---
            result["trajectory"] = {
                "states":     sim["states"].tolist(),
                "controls":   sim["controls"].tolist(),
                "references": sim["references"].tolist(),
                "t_vec":      sim["t_vec"].tolist(),
            }

            print(f"  Solve: mean={result['solve_time_mean_ms']:.3f}ms  "
                  f"std={result['solve_time_std_ms']:.3f}ms  "
                  f"max={result['solve_time_max_ms']:.3f}ms")
            print(f"  Code: {result['code_lines']} lines  "
                  f"{result['code_bytes']/1024:.1f} kB  "
                  f"binary={lib_b/1024:.1f} kB"
                  + (f"  (+{result['runtime_deps_bytes']/1024:.0f} kB runtime)"
                     if result.get('runtime_deps_bytes', 0) > 0 else ""))

            result["status"] = "ok"

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            result["status"]       = "error"
            result["error_message"] = str(e)

        all_results["solvers"][sname] = result
        print()

    # Save results
    out_path = os.path.join(results_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")

    return all_results


def run_scaling_benchmark(
    problem_name: str = "inverted_pendulum",
    solvers: list = None,
    results_dir: str = None,
    horizons: list[int] | None = None,
    skip_codegen: bool = False,
) -> dict:
    """Run a horizon sweep in fresh subprocesses and store timing/memory scaling results."""
    if solvers is None:
        solvers = list(SOLVER_REGISTRY.keys())
    solvers = list(solvers)
    if horizons is None:
        horizons = [10, 20, 30, 50, 80, 120]
    if results_dir is None:
        results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    scaling = {
        "problem": problem_name,
        "solvers": solvers,
        "horizons": horizons,
        "runs": [],
        "method": "fresh-subprocess-per-horizon",
    }

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"  Scaling sweep: Np=Nc={horizon}")
        print(f"{'='*60}\n")
        output_name = f"benchmark_results_N{horizon}.json"
        cmd = [
            sys.executable,
            os.path.join(_ROOT, "benchmark.py"),
            "--problem", problem_name,
            "--results-dir", results_dir,
            "--output-name", output_name,
            "--Np", str(horizon),
            "--Nc", str(horizon),
            "--solvers", *solvers,
        ]
        if skip_codegen:
            cmd.append("--skip-codegen")
        subprocess.run(cmd, check=True)
        with open(os.path.join(results_dir, output_name)) as f:
            result = json.load(f)
        scaling["runs"].append({
            "Np": horizon,
            "Nc": horizon,
            "results_file": output_name,
            "results": result,
        })

    out_path = os.path.join(results_dir, "scaling_results.json")
    with open(out_path, "w") as f:
        json.dump(scaling, f, indent=2)
    print(f"Scaling results saved to {out_path}")
    return scaling


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MPC benchmark")
    parser.add_argument("--problem",  default="inverted_pendulum",
                        help="Problem name (inverted_pendulum, double_integrator)")
    parser.add_argument("--solvers",  nargs="+",
                        default=["lmpc", "casadi", "cvxpygen", "acados"],
                        help="Solvers to benchmark")
    parser.add_argument("--skip-codegen", action="store_true",
                        help="Skip code generation (reuse existing)")
    parser.add_argument("--Np", type=int, default=None,
                        help="Override prediction horizon")
    parser.add_argument("--Nc", type=int, default=None,
                        help="Override control horizon")
    parser.add_argument("--scaling", action="store_true",
                        help="Run horizon-scaling sweep instead of one benchmark")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[50, 75, 100, 125],
                        help="Horizons to use for --scaling")
    parser.add_argument("--results-dir", default=os.path.join(_ROOT, "results"),
                        help="Directory for results JSON")
    parser.add_argument("--output-name", default="benchmark_results.json",
                        help="Output JSON file name inside --results-dir")
    args = parser.parse_args()

    overrides = {}
    if args.Np is not None:
        overrides["Np"] = args.Np
    if args.Nc is not None:
        overrides["Nc"] = args.Nc

    if args.scaling:
        run_scaling_benchmark(
            problem_name=args.problem,
            solvers=args.solvers,
            results_dir=args.results_dir,
            skip_codegen=args.skip_codegen,
            horizons=args.horizons,
        )
    else:
        run_benchmark(
            problem_name=args.problem,
            solvers=args.solvers,
            results_dir=args.results_dir,
            skip_codegen=args.skip_codegen,
            problem_overrides=overrides or None,
            output_name=args.output_name,
        )
