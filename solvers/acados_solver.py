"""
acados solver wrapper for linear MPC with C code generation.

All five solvers solve the SAME augmented-state increment (Δu) OCP:

  min  Σ_{k=0}^{N-1} (1/2)||Ca ξ_k − r||²_Q  +  (1/2)||Δu_k||²_Rr
        +  (1/2)||Ca ξ_N − r||²_Q
  s.t. ξ_{k+1} = Aa ξ_k + Ba Δu_k
       ξ[nx:] ∈ [u_lb, u_ub]   (u_prev part of augmented state bounded)
       ξ_0 = [x_current; u_prev]

where ξ = [x; u_prev], Δu is the control increment, and
  Aa = [[Ad, Bd],[0, I]],  Ba = [[Bd],[I]],  Ca = [C | 0].
"""

import os
import shutil


def _first_existing_dir(candidates: list[str], required_paths: list[str] | None = None) -> str | None:
    required_paths = required_paths or []
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isdir(candidate) and all(os.path.exists(os.path.join(candidate, rel)) for rel in required_paths):
            return candidate
    return None


def _resolve_acados_template_dir() -> str:
    source_dir = _first_existing_dir(
        [
            os.environ.get("ACADOS_SOURCE_DIR"),
            os.environ.get("ACADOS_ROOT"),
            os.environ.get("ACADOS_PATH"),
            "/tmp/acados",
            "/tmp/acados_install",
        ],
        required_paths=["interfaces/acados_template/acados_template/c_templates_tera"],
    )
    if source_dir is None:
        raise RuntimeError(
            "Could not locate an acados source tree. Set ACADOS_SOURCE_DIR or install acados "
            "under /tmp/acados."
        )
    return source_dir


def _resolve_acados_lib_dir(template_dir: str) -> str:
    lib_dir = _first_existing_dir(
        [
            os.environ.get("ACADOS_LIB_PATH"),
            os.path.join(os.environ.get("ACADOS_INSTALL_DIR", ""), "lib"),
            "/tmp/acados_install/lib",
            os.path.join(template_dir, "lib"),
        ],
        required_paths=["libacados.so"],
    )
    if lib_dir is None:
        raise RuntimeError(
            "Could not locate the acados shared libraries. Set ACADOS_LIB_PATH or "
            "ACADOS_INSTALL_DIR, or install them under /tmp/acados_install/lib."
    )
    return lib_dir


def _ensure_codegen_metadata(template_dir: str, lib_dir: str) -> str:
    install_root = os.path.dirname(lib_dir)
    source_lib_dir = os.path.join(template_dir, "lib")
    for fname in ("link_libs.json", "git_commit_hash"):
        src = os.path.join(source_lib_dir, fname)
        dst = os.path.join(lib_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    template_src = os.path.join(
        template_dir, "interfaces", "acados_template", "acados_template", "c_templates_tera"
    )
    template_dst = os.path.join(
        install_root, "interfaces", "acados_template", "acados_template", "c_templates_tera"
    )
    if os.path.isdir(template_src) and not os.path.exists(template_dst):
        os.makedirs(os.path.dirname(template_dst), exist_ok=True)
        os.symlink(template_src, template_dst)

    tera_src = os.path.join(template_dir, "bin", "t_renderer")
    tera_dst = os.path.join(install_root, "bin", "t_renderer")
    if os.path.isfile(tera_src) and not os.path.exists(tera_dst):
        os.makedirs(os.path.dirname(tera_dst), exist_ok=True)
        os.symlink(tera_src, tera_dst)

    return install_root


_ACADOS_TEMPLATE_DIR = _resolve_acados_template_dir()
_ACADOS_LIB_DIR = _resolve_acados_lib_dir(_ACADOS_TEMPLATE_DIR)
_ACADOS_INSTALL_ROOT = _ensure_codegen_metadata(_ACADOS_TEMPLATE_DIR, _ACADOS_LIB_DIR)
_ACADOS_SOURCE_DIR = _first_existing_dir(
    [
        os.environ.get("ACADOS_SOURCE_DIR"),
        _ACADOS_INSTALL_ROOT,
        _ACADOS_TEMPLATE_DIR,
    ],
    required_paths=["include", "interfaces/acados_template/acados_template/c_templates_tera"],
)
if _ACADOS_SOURCE_DIR is None:
    raise RuntimeError("Could not assemble a usable acados source/include/template layout.")

os.environ["ACADOS_SOURCE_DIR"] = _ACADOS_SOURCE_DIR
os.environ["TERA_PATH"] = os.environ.get("TERA_PATH") or os.path.join(_ACADOS_SOURCE_DIR, "bin", "t_renderer")
os.environ["LD_LIBRARY_PATH"] = _ACADOS_LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import ctypes

for _lib in ['libblasfeo.so', 'libhpipm.so', 'libqpOASES_e.so', 'libosqp.so', 'libacados.so']:
    _lib_path = os.path.join(_ACADOS_LIB_DIR, _lib)
    if os.path.isfile(_lib_path):
        try:
            ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import time
import numpy as np

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    import casadi as ca
    _ACADOS_AVAILABLE = True
except ImportError as _acados_import_error:
    _ACADOS_AVAILABLE = False


def _create_acados_ocp(prob: dict, codegen_dir: str) -> "AcadosOcp":
    """Build the acados OCP for the augmented-state Δu MPC formulation."""
    nx, nu, ny, na = prob['nx'], prob['nu'], prob['ny'], prob['na']
    Aa, Ba, Ca     = prob['Aa'], prob['Ba'], prob['Ca']
    Q, Rr          = prob['Q'], prob['Rr']
    Np             = prob['Np']
    u_lb, u_ub     = prob['u_lb'], prob['u_ub']

    # --- Linear discrete model (augmented state ξ=[x;u_prev], increment Δu as input) ---
    model      = AcadosModel()
    model.name = 'acados_aug'

    xi_sym = ca.MX.sym('xi', na)
    du_sym = ca.MX.sym('du', nu)

    model.x = xi_sym
    model.u = du_sym
    model.disc_dyn_expr = ca.MX(Aa) @ xi_sym + ca.MX(Ba) @ du_sym

    # --- OCP ---
    ocp       = AcadosOcp()
    ocp.model = model
    ocp.dims.N = Np
    ocp.code_gen_opts.acados_lib_path = _ACADOS_LIB_DIR
    
    # This is a strictly linear MPC problem, so use acados' linear least-squares
    # cost instead of the generic nonlinear-LS path.
    ocp.cost.cost_type   = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # Match the condensed solvers exactly:
    # each stage k penalises y_{k+1} = Ca * xi_{k+1} and Δu_k.
    # Since xi_{k+1} = Aa * xi_k + Ba * Δu_k, this is still a linear-LS cost:
    #   y_stage = [Ca*Aa * xi_k + Ca*Ba * du_k;
    #              du_k]
    ocp.cost.yref   = np.zeros(ny + nu)   # updated each solve
    ocp.cost.yref_e = np.zeros(ny)

    ocp.cost.Vx = np.zeros((ny + nu, na))
    ocp.cost.Vx[:ny, :] = Ca @ Aa
    ocp.cost.Vu = np.zeros((ny + nu, nu))
    ocp.cost.Vu[:ny, :] = Ca @ Ba
    ocp.cost.Vu[ny:, :] = np.eye(nu)
    # No separate terminal cost: the condensed solvers already include y_N exactly
    # once as the last stage term (k = N-1).
    ocp.cost.Vx_e = Ca.copy()

    # Stage weight: blkdiag(Q, Rr)
    ocp.cost.W   = np.diag(np.concatenate([Q, Rr]))   # (ny+nu) × (ny+nu)
    ocp.cost.W_e = np.zeros((ny, ny))

    # Initial augmented-state equality constraint (set per-solve)
    ocp.constraints.x0 = np.zeros(na)

    # Mixed linear constraints on the predicted next output y_{k+1} and applied input u_k:
    #
    #   y_{k+1} = Ca * Aa * xi_k + Ca * Ba * du_k
    #   u_k     = [0 I] * xi_k + I * du_k
    #
    # This matches the sparse/condensed formulations used by the other solvers.
    S_u = np.hstack([np.zeros((nu, nx)), np.eye(nu)])
    Cg = np.vstack([Ca @ Aa, S_u])
    Dg = np.vstack([Ca @ Ba, np.eye(nu)])

    ocp.constraints.C = Cg
    ocp.constraints.D = Dg
    ocp.constraints.lg = np.concatenate([prob["y_lb"], u_lb]).astype(float)
    ocp.constraints.ug = np.concatenate([prob["y_ub"], u_ub]).astype(float)

    # No terminal mixed constraints are needed: stage N-1 already constrains y_N and u_{N-1}.
    ocp.constraints.C_e = np.zeros((0, na))
    ocp.constraints.lg_e = np.zeros(0)
    ocp.constraints.ug_e = np.zeros(0)

    # Soft constraints for output bounds only (same behaviour as the other solvers).
    soft_w = float(prob.get("soft_weight", 0))
    if soft_w > 0:
        idxsg = np.arange(ny, dtype=np.int32)
        ocp.constraints.idxsg = idxsg
        ocp.constraints.lsg = np.zeros(ny)
        ocp.constraints.usg = np.zeros(ny)

        # acados uses 0.5 * Z * s^2 while the condensed solvers use soft_w * s^2.
        # Use Z = 2 * soft_w so the penalties match.
        ocp.cost.Zl = 2 * soft_w * np.ones(ny)
        ocp.cost.Zu = 2 * soft_w * np.ones(ny)
        ocp.cost.zl = np.zeros(ny)
        ocp.cost.zu = np.zeros(ny)

    # Solver options
    qp_solver_name = prob.get("acados_qp_solver", "PARTIAL_CONDENSING_HPIPM")
    ocp.solver_options.qp_solver       = qp_solver_name
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.tf              = prob['Ts'] * Np
    ocp.solver_options.print_level     = 0
    ocp.solver_options.qp_solver_tol_stat = 1e-9
    ocp.solver_options.qp_solver_tol_eq   = 1e-9
    ocp.solver_options.qp_solver_tol_ineq = 1e-9
    ocp.solver_options.qp_solver_tol_comp = 1e-9

    ocp.code_gen_opts.code_export_directory = codegen_dir

    return ocp


class AcadosSolver:
    name = "acados"

    def __init__(self, prob: dict):
        self._prob       = prob
        self._solver     = None
        self._available  = _ACADOS_AVAILABLE
        self._codegen_dir = None

        if not _ACADOS_AVAILABLE:
            print(f"[AcadosSolver] acados not available: {_acados_import_error}")

    # ------------------------------------------------------------------
    def setup(self, codegen_dir: str = "codegen/acados_codegen") -> dict:
        """Generate C code, compile, and warm-up solve."""
        if not _ACADOS_AVAILABLE:
            self._available = False
            raise RuntimeError("acados_template is not importable.")

        self._codegen_dir = codegen_dir
        os.makedirs(codegen_dir, exist_ok=True)

        t0 = time.perf_counter()
        try:
            ocp = _create_acados_ocp(self._prob, codegen_dir)
            json_file = os.path.join(codegen_dir, 'acados_aug_acados_ocp.json')
            self._solver = AcadosOcpSolver(ocp, json_file=json_file, build=True, verbose=False)
        except Exception as exc:
            self._available = False
            raise RuntimeError(f"acados setup failed: {exc}") from exc

        setup_time = time.perf_counter() - t0

        prob = self._prob
        try:
            self.solve(prob['x0_sim'], prob['r_sim'], np.zeros(prob['nu']))
        except Exception:
            pass

        return {'codegen_dir': codegen_dir, 'setup_time_s': setup_time}

    # ------------------------------------------------------------------
    def solve(self, x0: np.ndarray, r: np.ndarray, u_prev: np.ndarray = None) -> np.ndarray:
        """
        Solve one MPC step.  Returns the absolute control u = u_prev + Δu*.
        """
        if not self._available or self._solver is None:
            raise RuntimeError("AcadosSolver is not available / not set up.")

        if u_prev is None:
            u_prev = np.zeros(self._prob['nu'])
        u_prev = np.atleast_1d(u_prev).flatten()

        prob   = self._prob
        nx, nu, ny, na = prob['nx'], prob['nu'], prob['ny'], prob['na']
        Np     = prob['Np']
        solver = self._solver

        # Initial augmented state ξ0 = [x0; u_prev]
        xi0 = np.concatenate([x0.flatten(), u_prev])
        solver.set(0, 'lbx', xi0)
        solver.set(0, 'ubx', xi0)

        # Stage and terminal cost references
        yref_stage = np.concatenate([r, np.zeros(nu)])   # [r; 0_du]
        yref_term  = r.copy()                             # ny output reference

        for k in range(Np):
            solver.set(k, 'yref', yref_stage)
        solver.set(Np, 'yref', yref_term)

        solver.solve()

        du_opt = solver.get(0, 'u')
        u_opt  = u_prev + du_opt
        return np.clip(u_opt, prob['u_lb'], prob['u_ub'])

    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        c_files = h_files = 0
        code_bytes = 0

        if self._codegen_dir and os.path.isdir(self._codegen_dir):
            for root, _dirs, files in os.walk(self._codegen_dir):
                for fname in files:
                    if fname.endswith(('.c', '.h')):
                        fpath = os.path.join(root, fname)
                        code_bytes += os.path.getsize(fpath)
                        if fname.endswith('.c'):
                            c_files += 1
                        else:
                            h_files += 1

        return {
            'code_lines':    _count_lines(self._codegen_dir),
            'code_bytes':    code_bytes,
            'library_bytes': _find_so_size(self._codegen_dir),
            'runtime_deps_bytes': _acados_runtime_bytes(),
            'solver_name':   'acados (HPIPM/SQP)',
            'notes':         f'{c_files} .c files, {h_files} .h files in {self._codegen_dir}',
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_so_size(directory: str) -> int:
    """Return size of the first libacados_ocp_solver*.so found."""
    if not directory or not os.path.isdir(directory):
        return 0
    for fname in os.listdir(directory):
        if fname.startswith('libacados_ocp_solver') and fname.endswith('.so'):
            return os.path.getsize(os.path.join(directory, fname))
    return 0


def _acados_runtime_bytes() -> int:
    """Sum the sizes of the acados runtime shared libraries (BLASFEO + HPIPM + acados)."""
    total = 0
    for lib in ['libblasfeo.so', 'libhpipm.so', 'libacados.so']:
        p = os.path.join(_ACADOS_LIB_DIR, lib)
        if os.path.isfile(p):
            total += os.path.getsize(p)
        else:
            # Try following symlinks
            for candidate in os.listdir(_ACADOS_LIB_DIR):
                if candidate.startswith(lib.replace('.so', '')):
                    fp = os.path.join(_ACADOS_LIB_DIR, candidate)
                    if os.path.isfile(fp) and not os.path.islink(fp):
                        total += os.path.getsize(fp)
                        break
    return total


def _count_lines(directory: str) -> int:
    if not directory or not os.path.isdir(directory):
        return 0
    total = 0
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(('.c', '.h')):
                try:
                    with open(os.path.join(root, fname), 'r', errors='replace') as f:
                        total += sum(1 for _ in f)
                except OSError:
                    pass
    return total


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from problem_definition import get_problem

    prob   = get_problem('inverted_pendulum')
    solver = AcadosSolver(prob)

    print("Setting up acados solver (augmented Δu formulation) …")
    info = solver.setup(codegen_dir='codegen/acados_codegen')
    print(f"  Setup done in {info['setup_time_s']:.1f} s")

    x0     = prob['x0_sim']
    r      = prob['r_sim']
    u_prev = np.zeros(prob['nu'])

    print("Running closed-loop simulation …")
    t_solves = []
    Ad, Bd = prob['Ad'], prob['Bd']
    for step in range(prob['n_steps']):
        t0    = time.perf_counter()
        u_opt = solver.solve(x0, r, u_prev)
        t_solves.append(time.perf_counter() - t0)

        x0     = Ad @ x0 + Bd @ u_opt
        u_prev = u_opt

    print(f"  Solved {prob['n_steps']} steps")
    print(f"  Mean solve time : {1e3*np.mean(t_solves):.3f} ms")
    print(f"  Max  solve time : {1e3*np.max(t_solves):.3f} ms")
    print(f"  Final state     : {x0}")

    metrics = solver.get_metrics()
    print(f"\nCode metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
