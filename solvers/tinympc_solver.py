"""
TinyMPC solver wrapper for linear MPC with C code generation.

TinyMPC is an ADMM-based QP solver designed for embedded systems.
This wrapper uses the augmented-state increment (Δu) formulation so that
all five solvers solve the SAME OCP:
  - Augmented state ξ = [x; u_prev] (na = nx+nu dimensions)
  - Input Δu (control increment)
  - Cost: Q_aug = Ca' diag(Q) Ca on ξ,  R_aug = diag(Rr) on Δu
  - State bounds include u_prev component ∈ [u_lb, u_ub]
  - Generates C++ (not pure C) with a small, self-contained solver.
"""

import glob
import os
import sys
import time
import numpy as np


class TinyMPCSolver:
    name = "tinympc"
    timing_repetitions = 16

    def __init__(self, prob: dict):
        self.prob = prob
        self._solver = None
        self._codegen_dir = None
        self._generated = None

    # ------------------------------------------------------------------
    def setup(self, codegen_dir: str = "codegen/tinympc_codegen") -> dict:
        """Set up TinyMPC on the augmented state, compute ADMM cache, generate C++ code."""
        import tinympc

        p = self.prob
        Aa, Ba, Ca = p["Aa"], p["Ba"], p["Ca"]
        nx, nu, ny, na = p["nx"], p["nu"], p["ny"], p["na"]
        Np = p["Np"]

        # --- Cost matrices on augmented state ---
        # Q_aug = Ca' diag(Q) Ca  (na × na)
        Q_aug = Ca.T @ np.diag(p["Q"]) @ Ca
        # Tiny regularisation on zero-diagonal entries for ADMM convergence
        for i in range(na):
            if Q_aug[i, i] < 1e-8:
                Q_aug[i, i] = 1e-4

        R_input = np.diag(p["Rr"])   # nu × nu  (Δu cost)

        # --- State bounds for augmented state ---
        # Pole-angle constraint on ξ[2] (same as x[2])
        x_min = np.full(na, -1e6)
        x_max = np.full(na,  1e6)
        # u_prev part of augmented state is bounded by input limits
        x_min[nx:] = p["u_lb"]
        x_max[nx:] = p["u_ub"]
        # Output (pole angle) bounds when soft_weight > 0
        if float(p.get("soft_weight", 0)) > 0:
            C_mat = p["C"]
            for i in range(ny):
                row = C_mat[i]
                nz_idx = np.where(np.abs(row) > 0.5)[0]
                if len(nz_idx) == 1 and abs(row[nz_idx[0]] - 1.0) < 1e-9:
                    j = nz_idx[0]
                    x_min[j] = max(x_min[j], p["y_lb"][i] / row[nz_idx[0]])
                    x_max[j] = min(x_max[j], p["y_ub"][i] / row[nz_idx[0]])

        # Δu is unconstrained (u bound is via augmented state)
        du_min = np.full(nu, -1e6)
        du_max = np.full(nu,  1e6)

        t0 = time.perf_counter()

        prob_tiny = tinympc.TinyMPC()
        prob_tiny.setup(
            Aa, Ba, Q_aug, R_input,
            N=Np,
            rho=1.0,
            x_min=x_min,
            x_max=x_max,
            u_min=du_min,
            u_max=du_max,
            abs_pri_tol=1e-4,
            abs_dua_tol=1e-4,
            max_iter=200,
            check_termination=10,
            en_state_bound=True,
            en_input_bound=True,
            verbose=False,
        )
        prob_tiny.compute_cache_terms()

        os.makedirs(codegen_dir, exist_ok=True)
        prob_tiny.codegen(codegen_dir, verbose=False)

        self._solver = prob_tiny
        self._codegen_dir = codegen_dir
        self._generated = _load_generated_module(codegen_dir)
        self._nx = nx
        self._nu = nu
        self._na = na
        self._Ca = Ca

        # Warm-up
        xi_ref = self._output_ref_to_aug_ref(p["r_sim"])
        xi0 = np.concatenate([p["x0_sim"], np.zeros(nu)])
        prob_tiny.set_x0(xi0)
        prob_tiny.set_x_ref(xi_ref)
        prob_tiny.solve()

        setup_time = time.perf_counter() - t0
        return {"codegen_dir": codegen_dir, "setup_time_s": setup_time}

    # ------------------------------------------------------------------
    def _output_ref_to_aug_ref(self, r: np.ndarray) -> np.ndarray:
        """
        Convert output reference r (ny,) to augmented-state reference ξ_ref (na,).
        Maps each output row of Ca to the corresponding state, zero elsewhere.
        """
        na = self._na
        Ca = self._Ca
        xi_ref = np.zeros(na)
        for i, ri in enumerate(r):
            row = Ca[i]
            nz_idx = np.where(np.abs(row) > 0.5)[0]
            if len(nz_idx) == 1 and abs(row[nz_idx[0]] - 1.0) < 1e-9:
                xi_ref[nz_idx[0]] = ri / row[nz_idx[0]]
        return xi_ref

    # ------------------------------------------------------------------
    def solve(self, x0: np.ndarray, r: np.ndarray, u_prev: np.ndarray = None) -> np.ndarray:
        """
        Compute the first optimal Δu and return u = u_prev + Δu*.

        Parameters
        ----------
        x0     : current plant state (nx,)
        r      : output reference (ny,)
        u_prev : previous applied input (nu,)

        Returns
        -------
        u_opt  : optimal absolute input (nu,)
        """
        if self._solver is None:
            raise RuntimeError("setup() must be called before solve().")

        if u_prev is None:
            u_prev = np.zeros(self._nu)
        u_prev = np.atleast_1d(u_prev).flatten()

        xi0   = np.concatenate([x0.flatten(), u_prev])
        xi_ref = self._output_ref_to_aug_ref(r)

        self._generated.set_x0(xi0.reshape(-1, 1))
        self._generated.set_x_ref(xi_ref.reshape(-1, 1))
        sol = self._generated.solve()
        du_opt = np.atleast_1d(np.array(sol["controls"], dtype=float).flatten())
        u_opt  = u_prev + du_opt
        return np.clip(u_opt, self.prob["u_lb"], self.prob["u_ub"])

    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        if self._codegen_dir is None:
            raise RuntimeError("setup() must be called before get_metrics().")

        extensions = (".c", ".h", ".cpp", ".hpp", ".cxx")
        SKIP_DIRS = {"build", "include"}
        all_files = []
        for root, _dirs, files in os.walk(self._codegen_dir):
            parts = set(root.split(os.sep))
            if parts & SKIP_DIRS:
                continue
            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    all_files.append(os.path.join(root, fname))

        code_lines = sum(
            sum(1 for _ in open(f, encoding="utf-8", errors="replace"))
            for f in all_files
        )
        code_bytes = sum(os.path.getsize(f) for f in all_files)

        # TinyMPC's CMake build produces a self-contained static library.
        lib_bytes = _find_file_size(self._codegen_dir, "libtinympcstatic.a")

        return {
            "code_lines":    code_lines,
            "code_bytes":    code_bytes,
            "library_bytes": lib_bytes,
            "solver_name":   "tinympc (ADMM)",
            "notes": (
                "Augmented-state (ξ=[x;u_prev]) Δu formulation. "
                "Q_aug = Ca'*diag(Q)*Ca on ξ, R on Δu. "
                "Hard bounds: pole angle + u_prev∈[u_lb,u_ub]. "
                "Generates C++ code (Eigen headers excluded from footprint)."
            ),
        }


def _find_file_size(root: str, filename: str) -> int:
    """Walk root and return the size of the first file matching filename, or 0."""
    if not root or not os.path.isdir(root):
        return 0
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.getsize(os.path.join(dirpath, filename))
    return 0


def _load_generated_module(codegen_dir: str):
    """Import the generated TinyMPC pybind module."""
    if not codegen_dir or not os.path.isdir(codegen_dir):
        raise RuntimeError("TinyMPC codegen directory missing.")
    if codegen_dir not in sys.path:
        sys.path.insert(0, codegen_dir)
    import tinympcgen
    return tinympcgen


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from problem_definition import get_problem

    prob   = get_problem("inverted_pendulum")
    solver = TinyMPCSolver(prob)

    print("Setting up TinyMPC solver (augmented Δu formulation) …")
    info = solver.setup(codegen_dir="codegen/tinympc_codegen")
    print(f"  Setup done in {info['setup_time_s']:.1f} s")

    x0     = prob["x0_sim"].copy()
    r      = prob["r_sim"].copy()
    u_prev = np.zeros(prob["nu"])
    Ad, Bd = prob["Ad"], prob["Bd"]

    times = []
    for step in range(prob["n_steps"]):
        t0 = time.perf_counter()
        u  = solver.solve(x0, r, u_prev)
        times.append(time.perf_counter() - t0)
        x0     = Ad @ x0 + (Bd @ u).flatten()
        u_prev = u.copy()

    times = np.array(times) * 1e3
    print(f"  Median solve: {np.median(times):.3f} ms")
    print(f"  Final state : {x0.round(4)}")

    metrics = solver.get_metrics()
    print(f"  Code: {metrics['code_lines']} lines  {metrics['code_bytes']/1024:.1f} kB")
