import time
import glob
import os
import ctypes
import numpy as np
from lmpc import MPC


class LMPCSolver:
    name = "lmpc"
    timing_repetitions = 32

    def __init__(self, prob: dict):
        self.prob = prob
        self._mpc = None
        self._codegen_dir = None
        self._lib = None
        self._control_buf = None

    def setup(self, codegen_dir: str = "codegen/lmpc_codegen") -> dict:
        """Create MPC object, warm up Julia JIT, and generate C code."""
        p = self.prob
        t0 = time.perf_counter()

        mpc = MPC(p["A"], p["B"], p["Ts"], C=p["C"], Nc=p["Nc"], Np=p["Np"])
        mpc.set_objective(Q=p["Q"].tolist(), R=p["R"].tolist(), Rr=p["Rr"].tolist())
        mpc.set_bounds(umin=p["u_lb"].tolist(), umax=p["u_ub"].tolist())
        if float(p.get("soft_weight", 0)) > 0:
            mpc.set_output_bounds(
                ymin=p["y_lb"].tolist(), ymax=p["y_ub"].tolist(), soft=True
            )

        # Warm-up solve (also triggers Julia JIT compilation)
        mpc.compute_control(x=p["x0_sim"].tolist(), r=p["r_sim"].tolist())

        os.makedirs(codegen_dir, exist_ok=True)
        mpc.codegen(dir=codegen_dir)
        self._lib = _build_and_load_codegen_lib(codegen_dir)
        self._control_buf = np.zeros(p["nu"], dtype=np.float64)

        setup_time = time.perf_counter() - t0
        self._mpc = mpc
        self._codegen_dir = codegen_dir

        return {"codegen_dir": codegen_dir, "setup_time_s": setup_time}

    def solve(self, x0: np.ndarray, r: np.ndarray, u_prev: np.ndarray = None) -> np.ndarray:
        """Compute control action using the generated C solver."""
        if self._lib is None:
            raise RuntimeError("setup() must be called before solve()")
        x0 = np.ascontiguousarray(np.atleast_1d(x0).astype(np.float64))
        r = np.ascontiguousarray(np.atleast_1d(r).astype(np.float64))
        disturbance = np.zeros(0, dtype=np.float64)
        exitflag = self._lib.mpc_compute_control(
            self._control_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            disturbance.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if exitflag <= 0:
            raise RuntimeError(f"lmpc generated solver failed with exitflag {exitflag}")
        return self._control_buf.copy()

    def get_metrics(self) -> dict:
        """Return code size metrics and solver metadata."""
        if self._codegen_dir is None:
            raise RuntimeError("setup() must be called before get_metrics()")

        c_files = glob.glob(os.path.join(self._codegen_dir, "**", "*.c"), recursive=True)
        h_files = glob.glob(os.path.join(self._codegen_dir, "**", "*.h"), recursive=True)
        all_files = c_files + h_files

        code_lines = sum(
            sum(1 for _ in open(f, encoding="utf-8", errors="replace")) for f in all_files
        )
        code_bytes = sum(os.path.getsize(f) for f in all_files)

        # Compile generated .c files to measure the actual binary (flash) footprint.
        # lmpc embeds DAQP directly in the generated code — no external runtime required.
        lib_bytes = _compile_c_files(self._codegen_dir)

        return {
            "solver_name": "lmpc (LinearMPC.jl / DAQP)",
            "codegen_dir": self._codegen_dir,
            "code_lines": code_lines,
            "code_bytes": code_bytes,
            "library_bytes": lib_bytes,
            "notes": (
                "Output-tracking MPC with move-blocking (Nc < Np), "
                "soft output constraints, input increment (ΔU) formulation."
            ),
        }


def _compile_c_files(codegen_dir: str) -> int:
    """Compile all .c files in codegen_dir and return total .o size in bytes."""
    import subprocess
    import tempfile
    if not codegen_dir or not os.path.isdir(codegen_dir):
        return 0
    c_files = glob.glob(os.path.join(codegen_dir, "*.c"))
    if not c_files:
        return 0
    total = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for src in c_files:
            obj = os.path.join(tmpdir, os.path.basename(src).replace(".c", ".o"))
            try:
                subprocess.run(
                    ["gcc", "-O2", "-c", src, "-o", obj],
                    capture_output=True, timeout=60
                )
                if os.path.exists(obj):
                    total += os.path.getsize(obj)
            except Exception:
                pass
    return total


def _build_and_load_codegen_lib(codegen_dir: str):
    """Build LMPC generated C into a shared library and load it with ctypes."""
    import subprocess
    if not codegen_dir or not os.path.isdir(codegen_dir):
        raise RuntimeError("LMPC codegen directory missing.")
    lib_path = os.path.join(codegen_dir, "liblmpc_codegen.so")
    c_files = glob.glob(os.path.join(codegen_dir, "*.c"))
    if not c_files:
        raise RuntimeError("No LMPC generated C files found.")
    cmd = ["gcc", "-O3", "-shared", "-fPIC", "-o", lib_path, *c_files, "-lm"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    lib = ctypes.CDLL(lib_path)
    lib.mpc_compute_control.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.mpc_compute_control.restype = ctypes.c_int
    return lib


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from problem_definition import get_problem

    prob = get_problem("inverted_pendulum")
    solver = LMPCSolver(prob)

    print("Setting up lmpc solver (includes Julia JIT warm-up + codegen)...")
    info = solver.setup(codegen_dir="codegen/lmpc_codegen")
    print(f"  setup_time : {info['setup_time_s']:.2f} s")
    print(f"  codegen_dir: {info['codegen_dir']}")

    # Benchmark solve time over n_steps
    x = prob["x0_sim"].copy()
    r = prob["r_sim"]
    Ad, Bd = prob["Ad"], prob["Bd"]
    n_steps = prob["n_steps"]

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u = solver.solve(x, r)
        times.append(time.perf_counter() - t0)
        x = Ad @ x + Bd.flatten() * u[0]

    times = np.array(times) * 1e3  # ms
    print(f"\nSolve timing over {n_steps} steps:")
    print(f"  mean : {times.mean():.3f} ms")
    print(f"  median: {np.median(times):.3f} ms")
    print(f"  min  : {times.min():.3f} ms")
    print(f"  max  : {times.max():.3f} ms")

    metrics = solver.get_metrics()
    print(f"\nCode metrics:")
    print(f"  solver_name: {metrics['solver_name']}")
    print(f"  code_lines : {metrics['code_lines']}")
    print(f"  code_bytes : {metrics['code_bytes']}")
    print(f"  notes      : {metrics['notes']}")
