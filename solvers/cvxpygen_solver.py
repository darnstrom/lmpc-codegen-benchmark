import glob
import os
import sys
import time

import cvxpy as cp
import numpy as np


class CvxpygenSolver:
    name = "cvxpygen"

    def __init__(self, prob: dict):
        self.prob = prob
        self._codegen_dir = None
        self._cvx_prob = None
        self._xi0_param = None
        self._r_param = None
        self._dU = None
        self._nu = None
        self._nx = None
        self._na = None

    def setup(self, codegen_dir: str = "codegen/cvxpygen_codegen") -> dict:
        """Define CVXPY sparse augmented-state Δu MPC, generate C code via cvxpygen."""
        import types
        if "pdaqp" not in sys.modules:
            stub = types.ModuleType("pdaqp")
            stub.MPQP = None
            sys.modules["pdaqp"] = stub
        from cvxpygen import cpg

        p = self.prob
        Aa, Ba, Ca = p["Aa"], p["Ba"], p["Ca"]
        nx, nu, ny, na = p["nx"], p["nu"], p["ny"], p["na"]
        Np = p["Np"]
        Q_vec, Rr_vec = p["Q"], p["Rr"]
        u_lb, u_ub = p["u_lb"], p["u_ub"]

        t0 = time.perf_counter()

        # --- Parameters (re-used at each solve call) ---
        # Augmented initial state ξ0 = [x0; u_prev]
        xi0_param = cp.Parameter(na, name="xi0")
        r_param   = cp.Parameter(ny, name="r")

        # --- Decision variables (sparse formulation over augmented state) ---
        Xi = cp.Variable((na, Np + 1), name="Xi")   # augmented state trajectory
        dU = cp.Variable((nu, Np),     name="dU")    # control increment trajectory

        Q_sqrt  = np.diag(np.sqrt(Q_vec))
        Rr_sqrt = np.diag(np.sqrt(Rr_vec))

        # --- Constraints ---
        constraints = [Xi[:, 0] == xi0_param]
        for k in range(Np):
            constraints.append(Xi[:, k + 1] == Aa @ Xi[:, k] + Ba @ dU[:, k])
            # u constraint via augmented state: ξ[nx:] = u
            constraints.append(Xi[nx:, k + 1] >= u_lb)
            constraints.append(Xi[nx:, k + 1] <= u_ub)

        # --- Cost ---
        # Stage costs for k=0..Np-1 using Xi[:,k+1] (indices 1..Np).
        # The terminal state Xi[:,Np] is already penalised by the stage cost at k=Np-1,
        # so no separate terminal term is needed (adding one would double-count it).
        cost = 0
        for k in range(Np):
            y_k = Ca @ Xi[:, k + 1]
            cost += cp.sum_squares(Q_sqrt @ (y_k - r_param))
            cost += cp.sum_squares(Rr_sqrt @ dU[:, k])

        # --- Soft output constraints (pole angle etc.) ---
        soft_w = float(p.get("soft_weight", 0))
        if soft_w > 0:
            y_lb_arr = p["y_lb"]
            y_ub_arr = p["y_ub"]
            s_lo = cp.Variable((ny, Np), name="s_lo", nonneg=True)
            s_hi = cp.Variable((ny, Np), name="s_hi", nonneg=True)
            cost += soft_w * (cp.sum_squares(s_lo) + cp.sum_squares(s_hi))
            for k in range(Np):
                y_k = Ca @ Xi[:, k + 1]
                constraints.append(y_k + s_lo[:, k] >= y_lb_arr)
                constraints.append(y_k - s_hi[:, k] <= y_ub_arr)

        cvx_prob = cp.Problem(cp.Minimize(cost), constraints)

        abs_codegen = os.path.abspath(codegen_dir)
        parent_dir  = os.path.dirname(abs_codegen)
        basename    = os.path.basename(abs_codegen)
        os.makedirs(parent_dir, exist_ok=True)

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        orig_dir = os.getcwd()
        try:
            os.chdir(parent_dir)
            cpg.generate_code(cvx_prob, code_dir=basename, solver="OSQP",
                              solver_opts={"eps_abs": 1e-7, "eps_rel": 1e-7,
                                           "max_iter": 20000, "warm_starting": True})
        finally:
            os.chdir(orig_dir)

        self._xi0_param = xi0_param
        self._r_param   = r_param
        self._dU        = dU
        self._Xi        = Xi
        self._cvx_prob  = cvx_prob
        self._codegen_dir = abs_codegen
        self._nx = nx
        self._nu = nu
        self._na = na

        # Warm-up: use x0_sim with zero u_prev
        xi0_init = np.concatenate([p["x0_sim"], np.zeros(nu)])
        xi0_param.value = xi0_init
        r_param.value   = p["r_sim"]
        cvx_prob.solve(method="CPG")

        setup_time = time.perf_counter() - t0
        return {"codegen_dir": abs_codegen, "setup_time_s": setup_time}

    def solve(self, x0: np.ndarray, r: np.ndarray,
              u_prev: np.ndarray = None) -> np.ndarray:
        """Return the first absolute control u = u_prev + Δu*."""
        if self._cvx_prob is None:
            raise RuntimeError("setup() must be called before solve()")

        if u_prev is None:
            u_prev = np.zeros(self._nu)
        u_prev = np.atleast_1d(u_prev).flatten()

        xi0 = np.concatenate([x0, u_prev])
        self._xi0_param.value = xi0
        self._r_param.value   = r
        self._cvx_prob.solve(method="CPG")

        du_opt = self._dU.value[:, 0]
        u_opt  = u_prev + np.atleast_1d(np.array(du_opt, dtype=float))
        return np.clip(u_opt, self.prob["u_lb"], self.prob["u_ub"])

    def get_metrics(self) -> dict:
        if self._codegen_dir is None:
            raise RuntimeError("setup() must be called before get_metrics()")

        c_files = glob.glob(os.path.join(self._codegen_dir, "**", "*.c"), recursive=True)
        h_files = glob.glob(os.path.join(self._codegen_dir, "**", "*.h"), recursive=True)
        all_files = c_files + h_files

        code_lines = sum(
            sum(1 for _ in open(f, encoding="utf-8", errors="replace"))
            for f in all_files
        )
        code_bytes = sum(os.path.getsize(f) for f in all_files)

        # Prefer the compiled static library produced by CVXPYgen's CMake build.
        lib_bytes = _find_file_size(self._codegen_dir, "libcpg.a")

        return {
            "solver_name":   "cvxpygen (OSQP)",
            "codegen_dir":   self._codegen_dir,
            "code_lines":    code_lines,
            "code_bytes":    code_bytes,
            "library_bytes": lib_bytes,
            "notes":         "Sparse augmented-state Δu QP (ξ=[x;u_prev] trajectory).",
        }


def _find_file_size(root: str, filename: str) -> int:
    """Walk root and return the size of the first file matching filename, or 0."""
    if not root or not os.path.isdir(root):
        return 0
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.getsize(os.path.join(dirpath, filename))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from problem_definition import get_problem

    prob   = get_problem("inverted_pendulum")
    solver = CvxpygenSolver(prob)

    print("Setting up cvxpygen solver …")
    info = solver.setup(codegen_dir="codegen/cvxpygen_codegen")
    print(f"  setup_time : {info['setup_time_s']:.2f} s")

    Ad, Bd = prob["Ad"], prob["Bd"]
    x  = prob["x0_sim"].copy()
    r  = prob["r_sim"]
    u_prev = np.zeros(prob["nu"])
    times = []
    for _ in range(prob["n_steps"]):
        t0 = time.perf_counter()
        u  = solver.solve(x, r, u_prev)
        times.append(time.perf_counter() - t0)
        x  = Ad @ x + (Bd @ u).flatten()
        u_prev = u

    times = np.array(times) * 1e3
    print(f"\n  Median solve: {np.median(times):.3f} ms")
    print(f"  Final state : {x.round(4)}")
    metrics = solver.get_metrics()
    print(f"  Code: {metrics['code_lines']} lines  {metrics['code_bytes']/1024:.1f} kB")
