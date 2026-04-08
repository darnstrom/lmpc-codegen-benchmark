"""
Generate comparison plots and a markdown report from benchmark results.
"""

import argparse
import json
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = os.path.dirname(os.path.abspath(__file__))

COLOURS = {
    "lmpc":     "#2196F3",   # blue
    "casadi":   "#4CAF50",   # green
    "cvxpygen": "#FF9800",   # orange
    "acados":   "#9C27B0",   # purple
    "tinympc":  "#F44336",   # red
}
DISPLAY_NAMES = {
    "lmpc":     "lmpc\n(DAQP)",
    "casadi":   "CasADi\n(qpOASES)",
    "cvxpygen": "cvxpygen\n(OSQP)",
    "acados":   "acados\n(HPIPM/SQP)",
    "tinympc":  "TinyMPC\n(ADMM)",
}


def _load_scaling_results():
    path = os.path.join(_ROOT, "results", "scaling_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _ok_solvers(results: dict) -> list:
    return [s for s, v in results["solvers"].items() if v.get("status") == "ok"]


def _fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_timing_boxplot(results: dict, out_dir: str) -> str:
    """Box plot of per-step solve times (ms) for all successful solvers."""
    solvers = _ok_solvers(results)
    if not solvers:
        return None

    fig, ax = _fig(7, 5)

    data    = []
    labels  = []
    colours = []
    for s in solvers:
        t = results["solvers"][s].get("solve_times_ms", [])
        if t:
            data.append(t)
            labels.append(DISPLAY_NAMES.get(s, s))
            colours.append(COLOURS.get(s, "#888"))

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for elem in ("whiskers", "caps", "fliers"):
        for line in bp[elem]:
            line.set_color("#555")

    ax.set_yscale("log")
    ax.set_ylabel("Solve time per step  [ms]", fontsize=11)
    ax.set_title("MPC solver: per-step solve time (closed-loop)", fontsize=12)
    ax.set_xticklabels(labels, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    path = os.path.join(out_dir, "timing_boxplot.png")
    _save(fig, path)
    return path


def plot_timing_bar(results: dict, out_dir: str) -> str:
    """Bar chart of median solve time with min/max error bars."""
    solvers = _ok_solvers(results)
    if not solvers:
        return None

    fig, ax = _fig(7, 4.5)

    medians = []
    mins    = []
    maxs    = []
    labels  = []
    colours = []

    for s in solvers:
        v = results["solvers"][s]
        med  = v.get("solve_time_median_ms", 0)
        mn   = v.get("solve_time_min_ms", 0)
        mx   = v.get("solve_time_max_ms", 0)
        medians.append(med)
        mins.append(med - mn)
        maxs.append(mx - med)
        labels.append(DISPLAY_NAMES.get(s, s))
        colours.append(COLOURS.get(s, "#888"))

    x = np.arange(len(solvers))
    bars = ax.bar(x, medians, color=colours, alpha=0.85, zorder=3,
                  error_kw=dict(ecolor="#333", capsize=5, elinewidth=1.5))
    ax.errorbar(x, medians, yerr=[mins, maxs], fmt="none",
                ecolor="#333", capsize=5, elinewidth=1.5, zorder=4)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Median solve time  [ms]", fontsize=11)
    ax.set_title("MPC solver: median solve time (log scale)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Annotate bars with values
    for xi, med in zip(x, medians):
        ax.text(xi, med * 1.4, f"{med:.3f}", ha="center", va="bottom",
                fontsize=8, color="#333")

    path = os.path.join(out_dir, "timing_bar.png")
    _save(fig, path)
    return path


def plot_memory_bar(results: dict, out_dir: str) -> str:
    """Bar chart comparing compiled binary size (kB), with runtime deps stacked for acados."""
    solvers = _ok_solvers(results)
    if not solvers:
        return None

    fig, ax = _fig(7, 4.5)

    binary_kb  = []
    runtime_kb = []
    labels     = []
    colours    = []

    for s in solvers:
        v = results["solvers"][s]
        bkb = v.get("binary_bytes", 0) / 1024
        rkb = v.get("runtime_deps_bytes", 0) / 1024
        binary_kb.append(bkb)
        runtime_kb.append(rkb)
        labels.append(DISPLAY_NAMES.get(s, s))
        colours.append(COLOURS.get(s, "#888"))

    x = np.arange(len(solvers))
    bars1 = ax.bar(x, binary_kb, color=colours, alpha=0.85, zorder=3,
                   label="Generated binary")
    bars2 = ax.bar(x, runtime_kb, bottom=binary_kb, color=colours, alpha=0.35,
                   hatch="///", zorder=3, label="Runtime libraries")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Compiled binary size  [kB]", fontsize=11)
    ax.set_title("Compiled binary footprint (flash / ROM, log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    for xi, bkb, rkb in zip(x, binary_kb, runtime_kb):
        total = bkb + rkb
        label = f"{bkb:.0f} kB" if rkb == 0 else f"{bkb:.0f}+{rkb:.0f} kB"
        ax.text(xi, total * 1.02, label, ha="center", va="bottom",
                fontsize=8, color="#333")

    if any(r > 0 for r in runtime_kb):
        ax.legend(fontsize=9)

    path = os.path.join(out_dir, "memory_bar.png")
    _save(fig, path)
    return path


def plot_code_lines_bar(results: dict, out_dir: str) -> str:
    """Bar chart comparing lines of generated C code."""
    solvers = _ok_solvers(results)
    if not solvers:
        return None

    fig, ax = _fig(7, 4.5)

    lines   = []
    labels  = []
    colours = []

    for s in solvers:
        v = results["solvers"][s]
        lines.append(v.get("code_lines", 0))
        labels.append(DISPLAY_NAMES.get(s, s))
        colours.append(COLOURS.get(s, "#888"))

    x = np.arange(len(solvers))
    ax.bar(x, lines, color=colours, alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Lines of generated C code", fontsize=11)
    ax.set_title("Generated code complexity (line count)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    for xi, n in zip(x, lines):
        ax.text(xi, n * 1.02, f"{n:,}", ha="center", va="bottom",
                fontsize=8, color="#333")

    path = os.path.join(out_dir, "code_lines_bar.png")
    _save(fig, path)
    return path


def plot_trajectories(results: dict, out_dir: str, prob_name: str) -> str:
    """Closed-loop state trajectories for all successful solvers."""
    from problem_definition import get_problem

    solvers = _ok_solvers(results)
    if not solvers:
        return None

    prob = get_problem(prob_name)
    is_invpend = "inverted_pendulum" in prob_name

    ncols = 2 if is_invpend else 1
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 if is_invpend else 7, 9), sharex=True)
    fig.suptitle("Closed-loop trajectories", fontsize=13, y=0.99)

    if ncols == 1:
        axes = np.array([[ax] for ax in axes])

    handles = []
    t_ref = None
    refs  = None

    # Plot lmpc last so its dashed line is always visible on top
    plot_order = [s for s in solvers if s != "lmpc"] + [s for s in solvers if s == "lmpc"]

    for s in plot_order:
        v   = results["solvers"][s]
        traj = v.get("trajectory")
        if traj is None:
            continue

        t  = np.array(traj["t_vec"])
        X  = np.array(traj["states"])
        U  = np.array(traj["controls"])
        R  = np.array(traj.get("references", []))

        col   = COLOURS.get(s, "#888")
        alpha = 0.90
        # lmpc uses a thicker dashed line so it stays visible when trajectories overlap
        ls    = "--"  if s == "lmpc" else "-"
        lw    = 2.0   if s == "lmpc" else 1.4
        zo    = 10    if s == "lmpc" else 2

        if is_invpend:
            axes[0, 0].plot(t, X[:, 0], color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo, label=s)
            axes[1, 0].plot(t, X[:, 1], color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo)
            axes[2, 0].plot(t[:-1], U[:, 0], color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo)
            axes[0, 1].plot(t, np.degrees(X[:, 2]), color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo)
            axes[1, 1].plot(t, np.degrees(X[:, 3]), color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo)
            axes[2, 1].set_visible(False)
        else:
            axes[0, 0].plot(t, X[:, 0], color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo, label=s)
            axes[1, 0].plot(t, X[:, 1], color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo)
            axes[2, 0].plot(t[:-1], U[:, 0], color=col, alpha=alpha, ls=ls, lw=lw, zorder=zo)

        # Use a line handle for lmpc to show the dashed style in the legend
        if s == "lmpc":
            import matplotlib.lines as mlines
            handles.append(mlines.Line2D([], [], color=col, ls=ls, lw=lw, label=s))
        else:
            handles.append(mpatches.Patch(color=col, label=s))

        # Save references from first successful solver (all should be same)
        if t_ref is None and len(R) > 0:
            t_ref = t[:-1]
            refs  = R

    if is_invpend:
        # Cart position: reference trajectory
        if refs is not None:
            axes[0, 0].plot(t_ref, refs[:, 0], color="k", ls="--", lw=1.2, label="reference")
            handles.append(mpatches.Patch(color="k", label="reference"))
        axes[0, 0].set_ylabel("Cart pos [m]")

        # Pole angle: constraint bounds (degrees)
        y_lb_deg = np.degrees(prob["y_lb"][1])
        y_ub_deg = np.degrees(prob["y_ub"][1])
        axes[0, 1].axhline(y_ub_deg, ls=":", c="#c62828", lw=1.2, label=f"bound ±{y_ub_deg:.0f}°")
        axes[0, 1].axhline(y_lb_deg, ls=":", c="#c62828", lw=1.2)
        axes[0, 1].set_ylabel("Pole angle [°]")
        handles.append(mpatches.Patch(color="#c62828", label=f"pole bound ±{y_ub_deg:.0f}°"))

        axes[1, 0].set_ylabel("Cart vel [m/s]")
        axes[1, 1].set_ylabel("Pole ang. vel [°/s]")

        # Force: input constraint bounds
        axes[2, 0].axhline(float(prob["u_ub"][0]), ls=":", c="#e65100", lw=1.2, label="input bound")
        axes[2, 0].axhline(float(prob["u_lb"][0]), ls=":", c="#e65100", lw=1.2)
        handles.append(mpatches.Patch(color="#e65100", label=f"input bound ±{prob['u_ub'][0]:.1f}"))
        axes[2, 0].set_ylabel("Force [N]")
        axes[2, 0].set_xlabel("Time [s]")
        axes[2, 1].set_xlabel("Time [s]")
    else:
        axes[0, 0].set_ylabel("Position [m]")
        axes[1, 0].set_ylabel("Velocity [m/s]")
        axes[2, 0].set_ylabel("Control [m/s²]")
        axes[2, 0].set_xlabel("Time [s]")

    for row_axes in axes:
        for ax in row_axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               ncol=1, fontsize=9, framealpha=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    path = os.path.join(out_dir, "trajectories.png")
    _save(fig, path)
    return path


def plot_timing_cumulative(results: dict, out_dir: str) -> str:
    """Cumulative distribution of solve times."""
    solvers = _ok_solvers(results)
    if not solvers:
        return None

    fig, ax = _fig(7, 4.5)

    for s in solvers:
        v = results["solvers"][s]
        t = sorted(v.get("solve_times_ms", []))
        if not t:
            continue
        cdf = np.arange(1, len(t) + 1) / len(t)
        ax.plot(t, cdf * 100, color=COLOURS.get(s, "#888"),
                label=s, linewidth=2, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Solve time [ms]", fontsize=11)
    ax.set_ylabel("Cumulative [%]", fontsize=11)
    ax.set_title("Cumulative distribution of solve times", fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    path = os.path.join(out_dir, "timing_cdf.png")
    _save(fig, path)
    return path


def plot_scaling_time(scaling: dict, out_dir: str) -> str | None:
    """Plot median solve time versus horizon."""
    runs = scaling.get("runs", [])
    if not runs:
        return None

    fig, ax = _fig(7, 4.5)
    solver_names = scaling.get("solvers", [])

    for s in solver_names:
        xs, ys = [], []
        for run in runs:
            v = run["results"]["solvers"].get(s, {})
            if v.get("status") != "ok":
                continue
            xs.append(run["Np"])
            ys.append(v.get("solve_time_median_ms"))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, color=COLOURS.get(s, "#888"), label=s)

    ax.set_xlabel("Prediction horizon $N_p$", fontsize=11)
    ax.set_ylabel("Median solve time [ms]", fontsize=11)
    ax.set_title("Solve time scaling with horizon (median)", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    path = os.path.join(out_dir, "scaling_time.png")
    _save(fig, path)
    return path


def plot_scaling_memory(scaling: dict, out_dir: str) -> str | None:
    """Plot total deployed bytes versus horizon."""
    runs = scaling.get("runs", [])
    if not runs:
        return None

    fig, ax = _fig(7, 4.5)
    solver_names = scaling.get("solvers", [])

    for s in solver_names:
        xs, ys = [], []
        for run in runs:
            v = run["results"]["solvers"].get(s, {})
            if v.get("status") != "ok":
                continue
            total = v.get("binary_bytes", 0) + v.get("runtime_deps_bytes", 0)
            xs.append(run["Np"])
            ys.append(total / 1024)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, color=COLOURS.get(s, "#888"), label=s)

    ax.set_xlabel("Prediction horizon $N_p$", fontsize=11)
    ax.set_ylabel("Total deployed footprint [kB]", fontsize=11)
    ax.set_title("Total deployed footprint vs horizon", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    path = os.path.join(out_dir, "scaling_memory.png")
    _save(fig, path)
    return path

RUNTIME_MODE = {
    "lmpc": "generated C via ctypes",
    "casadi": "Python qpsol wrapper",
    "cvxpygen": "generated solver via CPG",
    "acados": "generated solver via acados shared lib",
    "tinympc": "generated pybind module",
}


def _table_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(headers, rows):
    sep = ["-" * max(len(h), 6) for h in headers]
    lines = [_table_row(headers), _table_row(sep)]
    for row in rows:
        lines.append(_table_row(row))
    return "\n".join(lines)

def main(results_path: str = None, plots_dir: str = None):
    if results_path is None:
        results_path = os.path.join(_ROOT, "results", "benchmark_results.json")
    if plots_dir is None:
        plots_dir = os.path.join(_ROOT, "results", "plots")

    with open(results_path) as f:
        results = json.load(f)

    scaling_path = os.path.join(os.path.dirname(results_path), "scaling_results.json")
    if os.path.exists(scaling_path):
        with open(scaling_path) as f:
            results["scaling"] = json.load(f)

    # Stash n_steps and Nc in results for report
    from problem_definition import get_problem
    prob = get_problem(results.get("problem", "inverted_pendulum"))
    results["n_steps"] = prob["n_steps"]
    results["Nc"] = prob["Nc"]

    print("\nGenerating plots...")
    plot_timing_boxplot(results, plots_dir)
    plot_timing_bar(results, plots_dir)
    plot_timing_cumulative(results, plots_dir)
    plot_memory_bar(results, plots_dir)
    plot_code_lines_bar(results, plots_dir)
    plot_trajectories(results, plots_dir, results.get("problem", ""))
    if results.get("scaling"):
        plot_scaling_time(results["scaling"], plots_dir)
        plot_scaling_memory(results["scaling"], plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark plots and reports from saved JSON results.")
    parser.add_argument(
        "--results-path",
        default=None,
        help="Path to a benchmark_results*.json file (defaults to results/benchmark_results.json).",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory where plot images are written (defaults to results/plots).",
    )
    args = parser.parse_args()
    main(results_path=args.results_path, plots_dir=args.plots_dir)
