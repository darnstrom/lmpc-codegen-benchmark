#!/usr/bin/env python3
"""
run_all.py  –  Full MPC benchmark pipeline.

Steps:
  1. Setup each solver (generates C code)
  2. Run closed-loop simulations and record timing + footprint

Usage:
    uv run python run_all.py [--problem NAME] [--solvers lmpc casadi ...]

To benchmark a different problem:
    1. Add a new function in problem_definition.py
    2. Run:  uv run python run_all.py --problem my_problem
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Run MPC code-generation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--problem",
        default="inverted_pendulum",
        help="Problem name defined in problem_definition.py "
             "(default: inverted_pendulum)",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["lmpc", "casadi", "cvxpygen", "acados", "tinympc"],
        metavar="SOLVER",
        help="Solvers to include (default: all five)",
    )
    parser.add_argument(
        "--skip-codegen",
        action="store_true",
        help="Skip code generation and reuse existing codegen directories",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(_ROOT, "results"),
        help="Directory for results JSON and plots",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Also run a horizon-scaling sweep",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[50, 75, 100, 125],
        help="Horizons used for --scaling",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MPC Code-Generation Benchmark")
    print("=" * 60)
    print(f"  Problem : {args.problem}")
    print(f"  Solvers : {', '.join(args.solvers)}")
    print(f"  Results : {args.results_dir}")
    if args.scaling:
        print(f"  Scaling : {', '.join(str(h) for h in args.horizons)}")
    print()

    # ── Step 1+2: run benchmark ──────────────────────────────────────────────
    from benchmark import run_benchmark, run_scaling_benchmark
    results = run_benchmark(
        problem_name=args.problem,
        solvers=args.solvers,
        results_dir=args.results_dir,
        skip_codegen=args.skip_codegen,
    )
    if args.scaling:
        run_scaling_benchmark(
            problem_name=args.problem,
            solvers=args.solvers,
            results_dir=args.results_dir,
            horizons=args.horizons,
            skip_codegen=args.skip_codegen,
        )

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
