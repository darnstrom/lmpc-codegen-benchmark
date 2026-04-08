"""
CasADi solver wrapper for linear MPC with C code generation.

Uses the augmented-state increment (Δu) condensed QP formulation so that all five
solvers in the benchmark solve the SAME optimisation problem:

  min_ΔU  ΔU' H_a ΔU  +  2 g_a(ξ0,r)' ΔU
  s.t.    u_lb ≤ u_prev + L_tri ΔU ≤ u_ub   (cumulative-u bound)
          (+ soft output constraints if enabled)

where  ξ0 = [x0; u_prev],  H_a = Ga' Q_bar Ga + Rr_bar  (constant),
       g_a = Ga' Q_bar (Fa ξ0 − R_tiled)  (affine in ξ0 and r).

Online solving uses qpOASES (fallback: OSQP).  C code generated via
CasADi's CodeGenerator with the qrqp solver.
"""

import os
import time
import warnings

import casadi as ca
import numpy as np


class CasadiSolver:
    name = "casadi"

    def __init__(self, prob: dict):
        self.prob = prob

        nx   = prob["nx"]
        nu   = prob["nu"]
        ny   = prob["ny"]
        na   = prob["na"]   # nx + nu  (augmented state dimension)
        Np   = prob["Np"]
        Nc   = prob["Nc"]

        self.nx, self.nu, self.ny, self.na = nx, nu, ny, na
        self.Np, self.Nc = Np, Nc
        self._use_soft = float(prob.get("soft_weight", 0)) > 0

        # Decision-variable size: Nc*nu (ΔU) + slack vars when soft
        self.nz  = Nc * nu + 2 * Np * ny if self._use_soft else Nc * nu
        # Parameter size: [ξ0 (na); r (ny)]
        self.np_ = na + ny

        # Pre-tile output bounds for soft constraints
        self.y_lb_bar = np.tile(prob["y_lb"], Np)
        self.y_ub_bar = np.tile(prob["y_ub"], Np)

        self._solver         = None
        self._codegen_solver = None
        self._codegen_dir    = None

    # ------------------------------------------------------------------
    def _build_symbolic_qp(self):
        """Return (z_sym, p_sym, f_expr, g_expr) for the augmented Δu condensed QP."""
        prob = self.prob
        nx, nu, ny, na = self.nx, self.nu, self.ny, self.na
        Np, Nc = self.Np, self.Nc

        H_qp_a = prob["H_qp_a"]  # Nc*nu × Nc*nu  constant augmented Hessian
        Fa_mat = prob["Fa"]       # Np*ny × na      free response (augmented)
        Ga_mat = prob["Ga"]       # Np*ny × Nc*nu   forced response (augmented)
        Q_bar  = prob["Q_bar"]    # Np*ny × Np*ny   stage + DARE terminal weights
        L_tri  = prob["L_tri"]    # Nc*nu × Nc*nu   cumulative-u matrix
        soft_w = float(prob.get("soft_weight", 0))

        nz  = self.nz
        np_ = self.np_

        z_sym = ca.MX.sym("dU", nz)
        p_sym = ca.MX.sym("p",  np_)

        xi0 = p_sym[:na]          # augmented initial state [x0; u_prev]
        r   = p_sym[na:]          # output reference

        dU  = z_sym[:Nc * nu]     # control increments

        H_ca    = ca.DM(H_qp_a)
        Fa_ca   = ca.DM(Fa_mat)
        Ga_ca   = ca.DM(Ga_mat)
        Q_ca    = ca.DM(Q_bar)
        GatQ    = Ga_ca.T @ Q_ca

        R_tiled  = ca.repmat(r, Np, 1)
        free_err = Fa_ca @ xi0 - R_tiled   # Fa ξ0 − R̃  (free response error)

        # Objective: ΔU' H_a ΔU + 2 g_a' ΔU
        obj = ca.dot(dU, H_ca @ dU) + 2.0 * ca.dot(GatQ @ free_err, dU)

        # Cumulative-u constraint: u_lb ≤ u_prev + L_tri ΔU ≤ u_ub
        L_ca   = ca.DM(L_tri)
        g_u    = L_ca @ dU                  # Nc*nu × 1

        if self._use_soft:
            s_lo = z_sym[Nc * nu: Nc * nu + Np * ny]
            s_hi = z_sym[Nc * nu + Np * ny:]
            obj += soft_w * (ca.dot(s_lo, s_lo) + ca.dot(s_hi, s_hi))
            Y_pred = Fa_ca @ xi0 + Ga_ca @ dU   # predicted outputs
            g2 = Y_pred + s_lo                   # ≥ y_lb
            g3 = -Y_pred + s_hi                  # ≥ -y_ub
            g_expr = ca.vertcat(g_u, g2, g3)
        else:
            g_expr = g_u

        return z_sym, p_sym, obj, g_expr

    # ------------------------------------------------------------------
    def setup(self, codegen_dir: str = "codegen/casadi_codegen") -> dict:
        t0 = time.perf_counter()

        z_sym, p_sym, obj, g_expr = self._build_symbolic_qp()

        Nc, Np, ny, nu = self.Nc, self.Np, self.ny, self.nu
        nz = self.nz

        qp_dict = {"x": z_sym, "p": p_sym, "f": obj, "g": g_expr}

        # ΔU is unconstrained (u bound is enforced via g_u constraint).
        # Slacks ≥ 0 when soft constraints are active.
        n_slack = 2 * Np * ny if self._use_soft else 0
        self._lbx = np.concatenate([np.full(Nc * nu, -np.inf), np.zeros(n_slack)])
        self._ubx = np.full(nz, np.inf)

        # Number of cumulative-u constraints (always present) + soft rows
        self._ng_u = Nc * nu
        self._ng2  = Np * ny if self._use_soft else 0
        self._ng3  = Np * ny if self._use_soft else 0

        # ------ primary solver: qpOASES (fallback OSQP) ------
        solver_name = "qpoases"
        solver_opts = {"print_time": False, "printLevel": "none",
                       "error_on_fail": False}
        try:
            self._solver = ca.qpsol("mpc_qp", "qpoases", qp_dict, solver_opts)
        except Exception:
            warnings.warn("qpOASES not available, falling back to OSQP.", stacklevel=2)
            solver_name = "osqp"
            self._solver = ca.qpsol("mpc_qp", "osqp", qp_dict,
                                    {"print_time": False, "error_on_fail": False})

        self._solver_name_str = solver_name

        # ------ code-generation solver: qrqp ------
        cg_solver = ca.qpsol("mpc_qp_cg", "qrqp", qp_dict,
                             {"print_time": False, "error_on_fail": False})
        self._codegen_solver = cg_solver

        # ------ generate C code ------
        os.makedirs(codegen_dir, exist_ok=True)
        cg = ca.CodeGenerator(
            "casadi_mpc",
            {"with_header": True, "cpp": False, "verbose": False},
        )
        cg.add(cg_solver)
        cg.generate(codegen_dir + "/")
        self._codegen_dir = codegen_dir

        setup_time = time.perf_counter() - t0
        return {"codegen_dir": codegen_dir, "setup_time_s": setup_time}

    # ------------------------------------------------------------------
    def solve(self, x0: np.ndarray, r: np.ndarray,
              u_prev: np.ndarray = None) -> np.ndarray:
        """Solve one MPC step; returns the absolute control u (not Δu)."""
        if self._solver is None:
            raise RuntimeError("Call setup() before solve().")

        if u_prev is None:
            u_prev = np.zeros(self.nu)
        u_prev = np.atleast_1d(u_prev).flatten()

        prob   = self.prob
        Nc, nu = self.Nc, self.nu

        # Augmented initial state ξ0 = [x0; u_prev]
        xi0   = np.concatenate([x0, u_prev])
        p_val = np.concatenate([xi0, r])

        # Cumulative-u constraint bounds (time-varying through u_prev)
        u_lb_rep  = np.tile(prob["u_lb"], Nc)
        u_ub_rep  = np.tile(prob["u_ub"], Nc)
        u_prev_rep = np.tile(u_prev,      Nc)
        lbg_u = u_lb_rep - u_prev_rep
        ubg_u = u_ub_rep - u_prev_rep

        if self._use_soft:
            lbg = np.concatenate([lbg_u, self.y_lb_bar, -self.y_ub_bar])
            ubg = np.concatenate([ubg_u,
                                  np.full(self._ng2, np.inf),
                                  np.full(self._ng3, np.inf)])
        else:
            lbg = lbg_u
            ubg = ubg_u

        sol   = self._solver(
            x0=np.zeros(self.nz),
            p=p_val,
            lbx=self._lbx,
            ubx=self._ubx,
            lbg=lbg,
            ubg=ubg,
        )

        z_opt  = np.array(sol["x"]).flatten()
        du_opt = z_opt[:nu]                        # first Δu
        u_opt  = u_prev + du_opt                   # actual control
        return np.clip(u_opt, prob["u_lb"], prob["u_ub"])

    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        import glob
        code_lines, code_bytes = 0, 0
        if self._codegen_dir is not None:
            all_files = (
                glob.glob(os.path.join(self._codegen_dir, "**", "*.c"), recursive=True)
                + glob.glob(os.path.join(self._codegen_dir, "**", "*.h"), recursive=True)
            )
            for f in all_files:
                code_bytes += os.path.getsize(f)
                with open(f, encoding="utf-8", errors="replace") as fh:
                    code_lines += sum(1 for _ in fh)

        # CasADi embeds qpOASES/qrqp in the generated .c — compile to measure binary size.
        lib_bytes = _compile_c_file(
            os.path.join(self._codegen_dir or "", "casadi_mpc.c")
        ) if self._codegen_dir else 0

        return {
            "code_lines":    code_lines,
            "code_bytes":    code_bytes,
            "library_bytes": lib_bytes,
            "solver_name":   f"casadi ({self._solver_name_str} / qrqp codegen)",
            "notes": (
                "C code generated via CasADi qrqp. "
                "Augmented-state (ξ=[x;u_prev]) condensed Δu QP (Nc=Np). "
                "On-line solving uses qpOASES (or OSQP fallback)."
            ),
        }


def _compile_c_file(path: str) -> int:
    """Compile a single .c file and return the compiled .o size in bytes."""
    import subprocess
    import tempfile
    if not path or not os.path.isfile(path):
        return 0
    with tempfile.TemporaryDirectory() as tmpdir:
        obj = os.path.join(tmpdir, "out.o")
        try:
            subprocess.run(
                ["gcc", "-O2", "-c", path, "-o", obj],
                capture_output=True, timeout=120
            )
            return os.path.getsize(obj) if os.path.exists(obj) else 0
        except Exception:
            return 0




if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from problem_definition import get_problem

    prob = get_problem("inverted_pendulum")
    solver = CasadiSolver(prob)

    print("Setting up solver …")
    info = solver.setup(codegen_dir="codegen/casadi_codegen")
    print(f"  setup_time : {info['setup_time_s']:.3f} s")

    x0 = prob["x0_sim"].copy()
    r  = prob["r_sim"].copy()

    print("Running 5 solve calls …")
    for i in range(5):
        u = solver.solve(x0, r)
        print(f"  step {i}: u = {u}")
        x0 = prob["Ad"] @ x0 + (prob["Bd"] @ u).flatten()

    metrics = solver.get_metrics()
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
