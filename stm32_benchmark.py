#!/usr/bin/env python3
"""
Helpers for preparing and summarizing on-target STM32 benchmark runs.

Typical workflow:
  1. Generate solver code with run_all.py / benchmark.py.
  2. Run `prepare` to discover generated sources and emit a compile helper.
  3. Integrate each solver into an STM32 project and record timings on target.
  4. Save `arm-none-eabi-size` output per solver.
  5. Run `summarize` to merge timings + size reports into JSON/CSV tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CODEGEN_ROOT = ROOT / "codegen"
DEFAULT_OUTPUT_DIR = ROOT / "stm32_benchmark"
DEFAULT_SOLVERS = ["lmpc", "casadi", "cvxpygen", "acados", "tinympc"]

FLASH_SECTION_PREFIXES = (
    ".isr_vector",
    ".text",
    ".rodata",
    ".init",
    ".fini",
    ".init_array",
    ".fini_array",
    ".preinit_array",
    ".ARM.extab",
    ".ARM.exidx",
    ".gcc_except_table",
    ".eh_frame",
)
RAM_SECTION_PREFIXES = (
    ".data",
    ".bss",
    ".heap",
    "._user_heap_stack",
    ".stack",
    ".noinit",
    ".ramfunc",
    ".sram",
)

SOLVER_NOTES = {
    "lmpc": (
        "Generated output is plain C and is usually the simplest candidate for bare-metal "
        "integration on STM32."
    ),
    "casadi": (
        "Generated output is self-contained C, but the generated qrqp code can be fairly large."
    ),
    "cvxpygen": (
        "Generated output typically includes C sources plus an OSQP-based runtime inside the "
        "generated tree; import the full generated subtree, not just the top-level files."
    ),
    "acados": (
        "Generated output depends on acados/BLASFEO/HPIPM runtime code. On STM32F411 this may "
        "require a heavily reduced static build or may simply be too large for practical use."
    ),
    "tinympc": (
        "Generated output is C++; use the C++ compiler path in your STM32 project and verify any "
        "required header-only dependencies from the generated tree are present."
    ),
}


@dataclass
class SolverArtifacts:
    solver: str
    codegen_dir: Path
    language: str
    sources: list[Path]
    headers: list[Path]
    include_dirs: list[Path]
    notes: str

    def to_dict(self) -> dict:
        return {
            "solver": self.solver,
            "codegen_dir": str(self.codegen_dir),
            "language": self.language,
            "sources": [str(path) for path in self.sources],
            "headers": [str(path) for path in self.headers],
            "include_dirs": [str(path) for path in self.include_dirs],
            "notes": self.notes,
        }


def _existing_solver_dirs(codegen_root: Path, solvers: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for solver in solvers:
        candidate = codegen_root / f"{solver}_codegen"
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def _detect_language(paths: list[Path]) -> str:
    if any(path.suffix in {".cpp", ".cxx", ".cc"} for path in paths):
        return "c++"
    return "c"


def _discover_include_dirs(files: list[Path]) -> list[Path]:
    include_dirs = {path.parent.resolve() for path in files}
    return sorted(include_dirs)


def discover_solver_artifacts(codegen_root: Path, solvers: list[str]) -> list[SolverArtifacts]:
    artifacts: list[SolverArtifacts] = []
    for codegen_dir in _existing_solver_dirs(codegen_root, solvers):
        solver = codegen_dir.name.removesuffix("_codegen")
        sources = sorted(
            path.resolve()
            for path in codegen_dir.rglob("*")
            if path.suffix in {".c", ".cpp", ".cxx", ".cc"}
        )
        headers = sorted(
            path.resolve()
            for path in codegen_dir.rglob("*")
            if path.suffix in {".h", ".hpp", ".hh", ".hxx"}
        )
        artifacts.append(
            SolverArtifacts(
                solver=solver,
                codegen_dir=codegen_dir.resolve(),
                language=_detect_language(sources),
                sources=sources,
                headers=headers,
                include_dirs=_discover_include_dirs(sources + headers),
                notes=SOLVER_NOTES.get(solver, ""),
            )
        )
    return artifacts


def _build_compile_script(
    output_path: Path,
    artifacts: list[SolverArtifacts],
    cpu: str,
    fpu: str,
    float_abi: str,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'TOOLCHAIN_PREFIX="${TOOLCHAIN_PREFIX:-arm-none-eabi}"',
        'case "$TOOLCHAIN_PREFIX" in',
        '  *-) ;;',
        '  *) TOOLCHAIN_PREFIX="${TOOLCHAIN_PREFIX}-" ;;',
        "esac",
        'CC="${TOOLCHAIN_PREFIX}gcc"',
        'CXX="${TOOLCHAIN_PREFIX}g++"',
        'BUILD_ROOT="${BUILD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/object_build}"',
        'EXTRA_CFLAGS="${EXTRA_CFLAGS:-}"',
        'EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"',
        'EXTRA_CFLAGS_ARRAY=()',
        'EXTRA_CXXFLAGS_ARRAY=()',
        'if [[ -n "$EXTRA_CFLAGS" ]]; then read -r -a EXTRA_CFLAGS_ARRAY <<< "$EXTRA_CFLAGS"; fi',
        'if [[ -n "$EXTRA_CXXFLAGS" ]]; then read -r -a EXTRA_CXXFLAGS_ARRAY <<< "$EXTRA_CXXFLAGS"; fi',
        'TARGET_SOLVER="${1:-all}"',
        "",
        "COMMON_FLAGS=(",
        f'  "-mcpu={cpu}"',
        '  "-mthumb"',
        f'  "-mfpu={fpu}"',
        f'  "-mfloat-abi={float_abi}"',
        '  "-ffunction-sections"',
        '  "-fdata-sections"',
        '  "-Os"',
        ")",
        "",
        'KNOWN_SOLVERS=("all"',
    ]

    for artifact in artifacts:
        lines.append(f'  "{artifact.solver}"')

    lines.extend(
        [
            ")",
            "",
            'if [[ ! " ${KNOWN_SOLVERS[*]} " =~ " ${TARGET_SOLVER} " ]]; then',
            '  echo "Unknown solver: $TARGET_SOLVER" >&2',
            "  exit 1",
            "fi",
            "",
            'mkdir -p "$BUILD_ROOT"',
            "",
        ]
    )

    for artifact in artifacts:
        compiler = "$CXX" if artifact.language == "c++" else "$CC"
        extra_flags = '"${EXTRA_CXXFLAGS_ARRAY[@]}"' if artifact.language == "c++" else '"${EXTRA_CFLAGS_ARRAY[@]}"'
        stdflag = "-std=c++17" if artifact.language == "c++" else "-std=c11"
        lines.extend(
            [
                f'if [[ "$TARGET_SOLVER" == "all" || "$TARGET_SOLVER" == "{artifact.solver}" ]]; then',
                f'  echo "Compiling {artifact.solver} generated sources..."',
                f'  mkdir -p "$BUILD_ROOT/{artifact.solver}"',
                "  SOLVER_FLAGS=(",
                f'    "{stdflag}"',
            ]
        )
        for include_dir in artifact.include_dirs:
            lines.append(f'    "-I{include_dir}"')
        lines.extend(
            [
                "  )",
            ]
        )
        if artifact.sources:
            for source in artifact.sources:
                obj_name = source.name.rsplit(".", 1)[0] + ".o"
                command = (
                    f'  {compiler} "${{COMMON_FLAGS[@]}}" "${{SOLVER_FLAGS[@]}}" {extra_flags} '
                    f'-c {shlex.quote(str(source))} -o "$BUILD_ROOT/{artifact.solver}/{obj_name}"'
                )
                lines.extend(
                    [
                        f'  echo "[compile] {artifact.solver} :: {source}"',
                        command,
                    ]
                )
        else:
            lines.append(f'  echo "No sources found for solver {artifact.solver}"')
        lines.extend(
            [
                "fi",
                "",
            ]
        )

    lines.append(
        'echo "Object compilation complete. Link these objects into your STM32 project to measure firmware size."'
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(output_path.stat().st_mode | stat.S_IXUSR)


def _write_timing_template(output_path: Path, solvers: list[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "solver",
                "timed_runs",
                "warmup_runs",
                "solve_mean_us",
                "solve_median_us",
                "solve_max_us",
                "solve_min_us",
                "notes",
            ],
        )
        writer.writeheader()
        for solver in solvers:
            writer.writerow(
                {
                    "solver": solver,
                    "timed_runs": "",
                    "warmup_runs": "",
                    "solve_mean_us": "",
                    "solve_median_us": "",
                    "solve_max_us": "",
                    "solve_min_us": "",
                    "notes": "",
                }
            )


def command_prepare(args: argparse.Namespace) -> int:
    codegen_root = Path(args.codegen_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = discover_solver_artifacts(codegen_root, args.solvers)
    manifest = {
        "board": {
            "mcu": args.mcu,
            "cpu": args.cpu,
            "fpu": args.fpu,
            "float_abi": args.float_abi,
        },
        "codegen_root": str(codegen_root),
        "solvers": [artifact.to_dict() for artifact in artifacts],
        "missing_solvers": [
            solver
            for solver in args.solvers
            if solver not in {artifact.solver for artifact in artifacts}
        ],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _build_compile_script(
        output_dir / "compile_codegen_objects.sh",
        artifacts,
        cpu=args.cpu,
        fpu=args.fpu,
        float_abi=args.float_abi,
    )
    _write_timing_template(output_dir / "timings_template.csv", args.solvers)

    print(f"Wrote {manifest_path}")
    print(f"Wrote {output_dir / 'compile_codegen_objects.sh'}")
    print(f"Wrote {output_dir / 'timings_template.csv'}")
    if manifest["missing_solvers"]:
        print("Missing codegen directories for:", ", ".join(manifest["missing_solvers"]))
    return 0


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _load_timings(path: Path) -> dict[str, dict]:
    timings: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            solver = (row.get("solver") or "").strip()
            if not solver:
                continue
            timings[solver] = {
                "timed_runs": _float_or_none(row.get("timed_runs")),
                "warmup_runs": _float_or_none(row.get("warmup_runs")),
                "solve_mean_us": _float_or_none(row.get("solve_mean_us")),
                "solve_median_us": _float_or_none(row.get("solve_median_us")),
                "solve_max_us": _float_or_none(row.get("solve_max_us")),
                "solve_min_us": _float_or_none(row.get("solve_min_us")),
                "notes": (row.get("notes") or "").strip(),
            }
    return timings


def _parse_size_default(text: str) -> dict[str, int] | None:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 6 and all(part.isdigit() for part in parts[:4]):
            text_bytes, data_bytes, bss_bytes = map(int, parts[:3])
            return {
                "flash_bytes": text_bytes + data_bytes,
                "ram_bytes": data_bytes + bss_bytes,
                "text_bytes": text_bytes,
                "data_bytes": data_bytes,
                "bss_bytes": bss_bytes,
            }
    return None


def _parse_size_sections(text: str) -> dict[str, int]:
    flash_bytes = 0
    ram_bytes = 0
    section_bytes: dict[str, int] = {}

    for line in text.splitlines():
        match = re.match(r"^([\w.$]+)\s+(\d+)(?:\s+0x[0-9a-fA-F]+)?\s*$", line.strip())
        if not match:
            continue
        name = match.group(1)
        size = int(match.group(2))
        section_bytes[name] = size
        if name.startswith(FLASH_SECTION_PREFIXES):
            flash_bytes += size
        if name.startswith(RAM_SECTION_PREFIXES):
            ram_bytes += size

    return {
        "flash_bytes": flash_bytes,
        "ram_bytes": ram_bytes,
        "sections": section_bytes,
    }


def _load_size_report(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_size_default(text)
    if parsed is not None:
        return parsed
    section_parsed = _parse_size_sections(text)
    if section_parsed["flash_bytes"] == 0 and section_parsed["ram_bytes"] == 0:
        return None
    return section_parsed


def _markdown_table(rows: list[dict]) -> str:
    headers = [
        "solver",
        "solve_mean_us",
        "solve_max_us",
        "flash_bytes",
        "ram_bytes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "" if row.get(header) is None else str(row.get(header))
                for header in headers
            )
            + " |"
        )
    return "\n".join(lines)


def command_summarize(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    timings_path = Path(args.timings).resolve()
    size_dir = Path(args.size_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timings = _load_timings(timings_path)

    rows: list[dict] = []
    for solver_info in manifest.get("solvers", []):
        solver = solver_info["solver"]
        size_report = _load_size_report(size_dir / f"{solver}{args.size_suffix}")
        timing_report = timings.get(solver, {})
        row = {
            "solver": solver,
            "language": solver_info.get("language"),
            "solve_mean_us": timing_report.get("solve_mean_us"),
            "solve_median_us": timing_report.get("solve_median_us"),
            "solve_max_us": timing_report.get("solve_max_us"),
            "solve_min_us": timing_report.get("solve_min_us"),
            "timed_runs": timing_report.get("timed_runs"),
            "warmup_runs": timing_report.get("warmup_runs"),
            "flash_bytes": None if size_report is None else size_report.get("flash_bytes"),
            "ram_bytes": None if size_report is None else size_report.get("ram_bytes"),
            "text_bytes": None if size_report is None else size_report.get("text_bytes"),
            "data_bytes": None if size_report is None else size_report.get("data_bytes"),
            "bss_bytes": None if size_report is None else size_report.get("bss_bytes"),
            "size_sections": None if size_report is None else size_report.get("sections"),
            "notes": " | ".join(
                value
                for value in [solver_info.get("notes", ""), timing_report.get("notes", "")]
                if value
            ),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            float("inf") if row.get("solve_mean_us") is None else row["solve_mean_us"],
            row["solver"],
        )
    )

    summary = {
        "manifest": str(manifest_path),
        "timings": str(timings_path),
        "size_dir": str(size_dir),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "solver",
                "language",
                "solve_mean_us",
                "solve_median_us",
                "solve_max_us",
                "solve_min_us",
                "timed_runs",
                "warmup_runs",
                "flash_bytes",
                "ram_bytes",
                "text_bytes",
                "data_bytes",
                "bss_bytes",
                "notes",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(_markdown_table(rows))
    print(f"\nWrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    return 0


# ---------------------------------------------------------------------------
# STM32 firmware build helpers
# ---------------------------------------------------------------------------

STM32_TEMPLATE_DIR = ROOT / "stm32"

# Solvers that can be built for bare-metal STM32F411 with the provided template.
# acados is excluded because it requires the full acados/BLASFEO/HPIPM runtime.
STM32_SUPPORTED_SOLVERS = ["lmpc", "casadi", "tinympc", "cvxpygen"]

# Maps solver name -> adapter file (relative to stm32/adapters/)
_ADAPTER_FILES = {
    "lmpc":     "lmpc_adapter.c",
    "casadi":   "casadi_adapter.c",
    "tinympc":  "tinympc_adapter.cpp",
    "cvxpygen": "cvxpygen_adapter.c",
}

# Solvers whose generated code contains C++ (affects project() languages)
_CPP_SOLVERS = {"tinympc"}


def _check_tool(name: str) -> bool:
    """Return True if *name* resolves on PATH."""
    return shutil.which(name) is not None


def _fmt_c_array(vals) -> str:
    """Format a sequence of floats as a C brace-initialiser, e.g. {1.0, 2.0}."""
    return "{" + ", ".join(f"{float(v):.17g}" for v in vals) + "}"


def _generate_problem_data_h(prob: dict) -> str:
    """
    Generate problem_data.h content from a problem dict (from problem_definition.py).

    The header defines:
      - Dimension macros (NX, NU, NY, NA, NC, NP_HORIZON)
      - Benchmark initial-condition macros (BENCH_X0_INIT, BENCH_R_INIT, …)
      - CasADi QP dimension macros (NZ_CASADI, NP_CASADI, NG_U, NG_SOFT, NG_CASADI)
      - CASADI_SOFT_LBG_ARRAY[] (if soft constraints are active)
      - TinyMPC reference-mapping macro (XI_REF_ROWS)
    """
    import numpy as np

    nx, nu, ny, na = prob["nx"], prob["nu"], prob["ny"], prob["na"]
    Nc, Np = prob["Nc"], prob["Np"]
    x0     = prob["x0_sim"]
    r      = prob["r_sim"]
    u_lb   = prob["u_lb"]
    u_ub   = prob["u_ub"]
    y_lb   = prob["y_lb"]
    y_ub   = prob["y_ub"]
    Ca     = prob["Ca"]          # (ny × na) output matrix for augmented state

    use_soft   = float(prob.get("soft_weight", 0)) > 0
    nz_casadi  = Nc * nu + (2 * Np * ny if use_soft else 0)
    np_casadi  = na + ny
    ng_u       = Nc * nu
    ng_soft    = 2 * Np * ny if use_soft else 0
    ng_casadi  = ng_u + ng_soft

    # TinyMPC: find which augmented-state index each output component maps to.
    # Row i of Ca is a unit row pointing at state index ref_rows[i].
    # A row is treated as a "unit row at index j" when |Ca[i,j]| > UNIT_ROW_THRESH
    # and |Ca[i,j] - 1| < UNIT_ROW_THRESH.
    UNIT_ROW_THRESH = 0.5
    ref_rows = []
    for i in range(ny):
        row = Ca[i]
        nz_idx = np.where(np.abs(row) > UNIT_ROW_THRESH)[0]
        if len(nz_idx) == 1 and abs(float(row[nz_idx[0]]) - 1.0) < UNIT_ROW_THRESH:
            ref_rows.append(int(nz_idx[0]))
        else:
            ref_rows.append(0)   # fallback -- may need manual correction

    lines = [
        "/* Generated by stm32_benchmark.py build -- do not edit by hand. */",
        "#ifndef PROBLEM_DATA_H",
        "#define PROBLEM_DATA_H",
        "",
        "/* ---- Dimensions ---- */",
        f"#define NX          {nx}",
        f"#define NU          {nu}",
        f"#define NY          {ny}",
        f"#define NA          {na}    /* NX + NU  (augmented state) */",
        f"#define NC          {Nc}    /* control horizon */",
        f"#define NP_HORIZON  {Np}    /* prediction horizon */",
        "",
        "/* ---- Benchmark initial conditions ---- */",
        f"#define BENCH_X0_INIT      {_fmt_c_array(x0)}",
        f"#define BENCH_R_INIT       {_fmt_c_array(r)}",
        f"#define BENCH_U0_INIT      {_fmt_c_array(np.zeros(nu))}",
        f"#define BENCH_X0_INIT_AUG  {_fmt_c_array(np.concatenate([x0, np.zeros(nu)]))}",
        "",
        "/* ---- Control bounds ---- */",
        f"#define BENCH_U_LB   {_fmt_c_array(u_lb)}",
        f"#define BENCH_U_UB   {_fmt_c_array(u_ub)}",
        "",
        "/* ---- CasADi qrqp QP dimensions ---- */",
        f"#define NZ_CASADI   {nz_casadi}   /* NC*NU + 2*NP_HORIZON*NY (slacks when soft) */",
        f"#define NP_CASADI   {np_casadi}   /* NA + NY */",
        f"#define NG_U        {ng_u}        /* NC*NU cumulative-u constraints */",
        f"#define NG_SOFT     {ng_soft}     /* 2*NP_HORIZON*NY soft rows (0 if disabled) */",
        f"#define NG_CASADI   {ng_casadi}   /* NG_U + NG_SOFT */",
    ]

    if use_soft and ng_soft > 0:
        # lbg for g2 rows: y_lb tiled Np times
        # lbg for g3 rows: -y_ub tiled Np times
        soft_lbg = list(np.tile(y_lb, Np)) + list(np.tile(-y_ub, Np))
        vals_str = ",\n    ".join(f"{float(v):.17g}" for v in soft_lbg)
        lines += [
            "",
            "/* ---- Soft-constraint lower bounds for CasADi ---- */",
            f"static const double CASADI_SOFT_LBG_ARRAY[{ng_soft}] = {{",
            f"    {vals_str}",
            "};",
        ]

    lines += [
        "",
        "/* ---- TinyMPC: map output reference r -> augmented reference xi_ref ---- */",
        "/* xi_ref[XI_REF_ROWS[i]] = r[i] for i in 0..NY-1; all other entries = 0 */",
        f"#define XI_REF_ROWS  {{{', '.join(str(i) for i in ref_rows)}}}",
        "",
        "#endif /* PROBLEM_DATA_H */",
        "",
    ]
    return "\n".join(lines)


def _generate_cmake_lists(
    template_dir: Path,
    solver: str,
    artifact: SolverArtifacts,
    build_dir: Path,
    cpu: str = "cortex-m4",
    fpu: str = "fpv4-sp-d16",
    float_abi: str = "hard",
) -> str:
    """
    Generate a complete CMakeLists.txt for building the benchmark firmware for
    one solver.  The generated file lives in *build_dir*; all template sources
    are referenced by absolute path so the build works from any working directory.
    """
    is_cpp       = solver in _CPP_SOLVERS
    project_lang = "C CXX" if is_cpp else "C"
    adapter_file = template_dir / "adapters" / _ADAPTER_FILES[solver]
    codegen_dir  = artifact.codegen_dir

    mcu_flags = (
        f"-mcpu={cpu} -mthumb "
        f"-mfpu={fpu} -mfloat-abi={float_abi}"
    )

    # ---- Common template sources (C, compiled for every solver) ----
    common_sources = [
        str(template_dir / "startup.c"),
        str(template_dir / "bench_main.c"),
        str(template_dir / "uart.c"),
    ]

    # ---- Solver-specific source discovery ----
    if solver == "lmpc":
        solver_sources_cmake = (
            f'file(GLOB SOLVER_SRCS "{codegen_dir}/*.c")\n'
        )
        extra_includes = [str(codegen_dir)]

    elif solver == "casadi":
        # CasADi generates casadi_mpc.c (and casadi_mpc.h) only.
        solver_sources_cmake = (
            f'set(SOLVER_SRCS "{codegen_dir}/casadi_mpc.c")\n'
        )
        extra_includes = [str(codegen_dir)]

    elif solver == "tinympc":
        # TinyMPC generates C++ sources under src/tinympc/ and Eigen in ext/.
        solver_sources_cmake = (
            f'file(GLOB_RECURSE SOLVER_SRCS\n'
            f'    "{codegen_dir}/src/tinympc/*.cpp"\n'
            f'    "{codegen_dir}/src/tinympc/*.c"\n'
            f')\n'
        )
        extra_includes = [
            str(codegen_dir / "include"),
            str(codegen_dir / "problem_data"),
            str(codegen_dir / "ext"),          # Eigen headers
            str(codegen_dir / "ext" / "eigen"),
        ]

    elif solver == "cvxpygen":
        # CVXPYgen embeds OSQP as C sources inside the generated tree.
        solver_sources_cmake = (
            f'file(GLOB_RECURSE SOLVER_SRCS "{codegen_dir}/*.c")\n'
            f'# Exclude Python C-extension glue files (e.g. cpg_ext.c) that contain\n'
            f'# Python.h includes and must not be compiled for bare-metal targets.\n'
            f'list(FILTER SOLVER_SRCS EXCLUDE REGEX ".*_ext\\\\.c$")\n'
        )
        extra_includes = [
            str(codegen_dir),
            str(codegen_dir / "include"),
        ]

    else:
        solver_sources_cmake = f'file(GLOB SOLVER_SRCS "{codegen_dir}/*.c")\n'
        extra_includes = [str(codegen_dir)]

    all_includes = [
        str(build_dir),            # problem_data.h lives here
        str(template_dir),         # bench_main.c, dwt_timer.h, uart.h, solver_adapter.h
    ] + extra_includes

    linker_script = str(template_dir / "STM32F411CEUX_FLASH.ld")

    # ---- C++ options (only for tinympc) ----
    cpp_options_block = ""
    if is_cpp:
        cpp_options_block = (
            "target_compile_options(bench.elf PRIVATE\n"
            "    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti>\n"
            ")\n"
        )

    lines = [
        "# Generated by stm32_benchmark.py build -- do not edit by hand.",
        "cmake_minimum_required(VERSION 3.18)",
        f"project(stm32_mpc_{solver} {project_lang})",
        "",
        "set(CMAKE_C_STANDARD   11)",
        "set(CMAKE_CXX_STANDARD 17)",
        f'set(MCU_FLAGS "{mcu_flags}")',
        "",
        "# ---- Solver-specific source discovery ----",
        solver_sources_cmake,
        "# ---- Executable ----",
        "add_executable(bench.elf",
        *[f'    "{s}"' for s in common_sources],
        f'    "{adapter_file}"',
        "    ${SOLVER_SRCS}",
        ")",
        "",
        "# ---- Include directories ----",
        "target_include_directories(bench.elf PRIVATE",
        *[f'    "{inc}"' for inc in all_includes],
        ")",
        "",
        "# ---- Preprocessor definitions ----",
        "target_compile_definitions(bench.elf PRIVATE",
        "    STM32F411xE",
        ")",
        "",
        "# ---- Compile options ----",
        "separate_arguments(MCU_FLAGS_LIST UNIX_COMMAND ${MCU_FLAGS})",
        "target_compile_options(bench.elf PRIVATE",
        "    ${MCU_FLAGS_LIST}",
        "    -Os",
        "    -ffunction-sections",
        "    -fdata-sections",
        "    -Wall",
        "    -Wno-unused-function",
        ")",
        cpp_options_block,
        "# ---- Link options ----",
        "target_link_options(bench.elf PRIVATE",
        "    ${MCU_FLAGS_LIST}",
        f'    -T"{linker_script}"',
        "    -Wl,--gc-sections",
        "    -specs=nosys.specs",
        "    -specs=nano.specs",
        "    -lc -lm -lnosys",
        ")",
        "",
        "# ---- Post-build: print size and generate .bin for flashing ----",
        "find_program(ARM_SIZE arm-none-eabi-size REQUIRED)",
        "find_program(ARM_OBJCOPY arm-none-eabi-objcopy REQUIRED)",
        "add_custom_command(TARGET bench.elf POST_BUILD",
        '    COMMAND ${ARM_SIZE} "$<TARGET_FILE:bench.elf>"',
        '    COMMENT "Binary size:"',
        ")",
        "add_custom_command(TARGET bench.elf POST_BUILD",
        '    COMMAND ${ARM_OBJCOPY} -O binary "$<TARGET_FILE:bench.elf>" bench.bin',
        '    COMMENT "Generating bench.bin"',
        ")",
        "",
    ]
    return "\n".join(lines)


def _run_cmd(cmd: list[str], cwd: Path, label: str) -> tuple[bool, str]:
    """Run a subprocess, stream output, return (success, combined_output)."""
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=300,
        )
        ok = result.returncode == 0
        if not ok:
            print(f"  [ERROR] {label} failed (exit {result.returncode}):")
            for line in result.stdout.splitlines():
                print(f"    {line}")
        return ok, result.stdout
    except FileNotFoundError:
        msg = f"Command not found: {cmd[0]}"
        print(f"  [ERROR] {msg}")
        return False, msg
    except subprocess.TimeoutExpired:
        msg = f"{label} timed out after 300 s"
        print(f"  [ERROR] {msg}")
        return False, msg


def _parse_size_output(output: str) -> dict:
    """Parse arm-none-eabi-size output into a dict of section sizes."""
    sizes: dict[str, int] = {}
    for line in output.splitlines():
        m = re.match(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+(\S+)", line)
        if m:
            sizes["text"]  = int(m.group(1))
            sizes["data"]  = int(m.group(2))
            sizes["bss"]   = int(m.group(3))
            sizes["flash"] = sizes["text"] + sizes["data"]
            sizes["ram"]   = sizes["data"] + sizes["bss"]
    return sizes


def command_build(args) -> int:
    """
    Build STM32F411 firmware for each supported solver.

    For every solver that has a codegen directory, this command:
      1. Creates stm32_benchmark/build/<solver>/
      2. Writes problem_data.h (generated from problem_definition.py)
      3. Writes CMakeLists.txt (references template files and solver sources)
      4. Runs cmake (with arm_none_eabi.cmake toolchain) + cmake --build
      5. Reports Flash/RAM size from arm-none-eabi-size
      6. Saves per-solver size reports for later use by the summarize command
    """
    # ---- Check prerequisites ----
    missing = []
    for tool in ("cmake", "arm-none-eabi-gcc"):
        if not _check_tool(tool):
            missing.append(tool)
    if missing:
        print(
            "[ERROR] Missing required tools: " + ", ".join(missing) + "\n"
            "Install an ARM embedded toolchain, e.g.:\n"
            "  sudo apt install cmake gcc-arm-none-eabi   (Debian/Ubuntu)\n"
            "  brew install arm-none-eabi-gcc             (macOS via Homebrew)"
        )
        return 1

    # ---- Load problem definition ----
    try:
        sys.path.insert(0, str(ROOT))
        from problem_definition import get_problem  # type: ignore
        import numpy  # noqa: F401 — scipy/numpy needed by get_problem
        prob = get_problem(getattr(args, "problem", "inverted_pendulum"))
    except Exception as exc:
        print(f"[ERROR] Could not load problem_definition.py: {exc}")
        return 1

    # ---- Discover solver artifacts (reuse existing logic) ----
    codegen_root = Path(getattr(args, "codegen_root", str(DEFAULT_CODEGEN_ROOT)))
    output_dir   = Path(getattr(args, "output_dir", str(DEFAULT_OUTPUT_DIR)))
    solvers      = [s for s in getattr(args, "solvers", STM32_SUPPORTED_SOLVERS)
                    if s in STM32_SUPPORTED_SOLVERS]

    artifacts = discover_solver_artifacts(codegen_root, solvers)
    if not artifacts:
        print(
            "[ERROR] No solver codegen directories found under "
            f"{codegen_root}.\n"
            "Run  uv run python run_all.py --solvers lmpc casadi tinympc  first."
        )
        return 1

    problem_data_h = _generate_problem_data_h(prob)

    build_root  = output_dir / "build"
    size_dir    = output_dir / "measurements"
    size_dir.mkdir(parents=True, exist_ok=True)

    toolchain_file = STM32_TEMPLATE_DIR / "arm_none_eabi.cmake"
    cpu        = getattr(args, "cpu", "cortex-m4")
    fpu        = getattr(args, "fpu", "fpv4-sp-d16")
    float_abi  = getattr(args, "float_abi", "hard")

    built: list[str] = []
    failed: list[str] = []

    for artifact in artifacts:
        solver = artifact.solver
        print(f"\n{'='*60}")
        print(f"Building {solver} ...")
        print(f"{'='*60}")

        solver_build_dir = build_root / solver
        cmake_build_dir  = solver_build_dir / "_cmake_build"
        solver_build_dir.mkdir(parents=True, exist_ok=True)
        cmake_build_dir.mkdir(parents=True, exist_ok=True)

        # Write generated files into solver_build_dir.
        (solver_build_dir / "problem_data.h").write_text(
            problem_data_h, encoding="utf-8"
        )
        cmake_txt = _generate_cmake_lists(
            STM32_TEMPLATE_DIR, solver, artifact, solver_build_dir,
            cpu=cpu, fpu=fpu, float_abi=float_abi,
        )
        (solver_build_dir / "CMakeLists.txt").write_text(cmake_txt, encoding="utf-8")

        # Configure.
        cmake_configure = [
            "cmake", "-S", str(solver_build_dir), "-B", str(cmake_build_dir),
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        ok, _ = _run_cmd(cmake_configure, solver_build_dir, f"cmake configure ({solver})")
        if not ok:
            failed.append(solver)
            continue

        # Build.
        ok, build_out = _run_cmd(
            ["cmake", "--build", str(cmake_build_dir), "--", "-j4"],
            solver_build_dir, f"cmake build ({solver})",
        )
        if not ok:
            failed.append(solver)
            continue

        # Save .bin and size report.
        bin_src = cmake_build_dir / "bench.bin"
        elf_src = cmake_build_dir / "bench.elf"
        if bin_src.exists():
            shutil.copy(bin_src, solver_build_dir / "bench.bin")

        size_report = ""
        if elf_src.exists():
            ok_sz, size_out = _run_cmd(
                ["arm-none-eabi-size", str(elf_src)],
                solver_build_dir, f"size ({solver})",
            )
            if ok_sz:
                size_report = size_out
                (size_dir / f"{solver}.size").write_text(size_report, encoding="utf-8")
                sizes = _parse_size_output(size_report)
                print(
                    f"  OK — Flash: {sizes.get('flash', '?')} bytes, "
                    f"RAM: {sizes.get('ram', '?')} bytes"
                )

        built.append(solver)

    print(f"\n{'='*60}")
    if built:
        print(f"Built successfully: {', '.join(built)}")
    if failed:
        print(f"Failed:            {', '.join(failed)}")
    print(f"Firmware binaries: {build_root}/")
    print(f"Size reports:      {size_dir}/")
    return 0 if not failed or built else 1


def command_run(args) -> int:
    """
    Full end-to-end STM32 benchmark: build → flash → capture UART → summarize.

    Prerequisites:
      • arm-none-eabi-gcc and cmake in PATH (for build step)
      • st-flash, openocd, or pyocd in PATH (for flash step)
      • pyserial installed  (pip install pyserial)  (for capture step)
      • An STM32F411 board connected via ST-LINK USB

    The UART output (USART2, PA2/PA3, 115 200 Bd) is captured automatically.
    For Nucleo-F411RE boards PA2/PA3 are wired to the ST-LINK virtual COM port.

    If flashing/capture tools are unavailable the script prints instructions
    for running those steps manually.
    """
    output_dir  = Path(getattr(args, "output_dir", str(DEFAULT_OUTPUT_DIR)))
    port        = getattr(args, "port", None)
    flasher     = getattr(args, "flasher", "auto")
    no_flash    = getattr(args, "no_flash", False)
    baud        = 115200
    capture_timeout = getattr(args, "timeout", 60)

    # ---- Step 1: Build ----
    print("=== Step 1: Building firmware ===")
    rc = command_build(args)
    if rc != 0:
        return rc

    if no_flash:
        print("\n--no-flash specified — skipping flash/capture step.")
        print(f"Firmware binaries are in {output_dir / 'build'}/")
        print("Flash each .bin manually and fill in the timing CSV, then run:")
        print(f"  uv run python stm32_benchmark.py summarize --output-dir {output_dir}")
        return 0

    # ---- Check flash + capture tools ----
    _flasher_cmd = _find_flasher(flasher)
    _has_serial  = _try_import_serial()

    if not _flasher_cmd:
        _print_manual_flash_instructions(output_dir)
        return 0

    if not _has_serial:
        print(
            "\n[WARNING] pyserial is not installed — cannot capture UART automatically.\n"
            "Install it with:  pip install pyserial\n"
            "Then re-run this command with  --port /dev/ttyACM0  (or the correct port).\n"
        )
        _print_manual_flash_instructions(output_dir)
        return 0

    if port is None:
        port = _autodetect_port()
        if port is None:
            print(
                "\n[ERROR] Could not auto-detect a serial port.\n"
                "Specify the port with  --port /dev/ttyACM0  (or equivalent)."
            )
            return 1

    # ---- Step 2: Flash + Capture for each solver ----
    print(f"\n=== Step 2: Flash + capture (port={port}, flasher={_flasher_cmd[0]}) ===")
    build_root = output_dir / "build"
    solvers    = [s for s in getattr(args, "solvers", STM32_SUPPORTED_SOLVERS)
                  if s in STM32_SUPPORTED_SOLVERS]

    timing_rows: list[dict] = []

    for solver in solvers:
        bin_path = build_root / solver / "bench.bin"
        if not bin_path.exists():
            print(f"\n  Skipping {solver}: bench.bin not found.")
            continue

        print(f"\n  --- {solver} ---")
        print(f"  Flashing {bin_path} ...")

        ok_flash = _flash_binary(bin_path, _flasher_cmd)
        if not ok_flash:
            print(f"  [ERROR] Flash failed for {solver}.")
            continue

        # Brief pause for the MCU to reset and start executing.
        time.sleep(1.5)

        print(f"  Capturing UART output on {port} (timeout={capture_timeout}s) ...")
        row = _capture_uart_result(port, baud, capture_timeout)
        if row is None:
            print(f"  [ERROR] No BENCH_START/BENCH_END received from {solver}.")
            continue

        row["solver"] = solver   # override with known name in case label differs
        timing_rows.append(row)
        print(
            f"  cycles: mean={row['cycles_mean']:,}  max={row['cycles_max']:,}  "
            f"median={row['cycles_median']:,}  "
            f"(@{int(row['sysclk_hz']) // 1_000_000} MHz)"
        )

    if not timing_rows:
        print("\n[WARNING] No timing data captured.")
        return 1

    # ---- Write timing CSV (compatible with the summarize command) ----
    timings_path = output_dir / "timings.csv"
    _write_timings_csv(timing_rows, timings_path)
    print(f"\nWrote {timings_path}")

    # ---- Step 3: Summarize ----
    print("\n=== Step 3: Summarize ===")
    args.timings  = str(timings_path)
    args.manifest = str(output_dir / "manifest.json")
    args.size_dir = str(output_dir / "measurements")
    args.size_suffix = ".size"
    args.output_dir  = str(output_dir)
    # If no manifest exists yet, run prepare first.
    if not Path(args.manifest).exists():
        args.codegen_root = str(DEFAULT_CODEGEN_ROOT)
        args.mcu = "STM32F411xE"
        command_prepare(args)
    return command_summarize(args)


# ---------------------------------------------------------------------------
# Flash / capture helpers
# ---------------------------------------------------------------------------

def _find_flasher(preference: str) -> list[str] | None:
    """Return the flash command template for the available tool, or None."""
    candidates: list[tuple[str, list[str]]] = [
        ("st-flash",
         ["st-flash", "write", "{bin}", "0x08000000"]),
        ("openocd",
         ["openocd", "-f", "board/st_nucleo_f4.cfg",
          "-c", "program {bin} verify reset exit"]),
        ("pyocd",
         ["pyocd", "flash", "--target", "stm32f411re", "{bin}"]),
    ]
    if preference != "auto":
        candidates = [(n, c) for (n, c) in candidates if n == preference] + \
                     [(n, c) for (n, c) in candidates if n != preference]
    for name, cmd_template in candidates:
        if _check_tool(name):
            return cmd_template
    return None


def _flash_binary(bin_path: Path, cmd_template: list[str]) -> bool:
    """Flash *bin_path* using *cmd_template* (with ``{bin}`` placeholder)."""
    cmd = [c.replace("{bin}", str(bin_path)) for c in cmd_template]
    ok, _ = _run_cmd(cmd, bin_path.parent, "flash")
    return ok


def _try_import_serial() -> bool:
    try:
        import serial  # noqa: F401
        return True
    except ImportError:
        return False


def _autodetect_port() -> str | None:
    """Try to find a likely ST-LINK virtual COM port."""
    import glob as _glob
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/tty.usbmodem*"):
        ports = sorted(_glob.glob(pattern))
        if ports:
            return ports[0]
    return None


def _capture_uart_result(port: str, baud: int, timeout: float) -> dict | None:
    """
    Open *port* and wait for the BENCH_START…BENCH_END block.
    Returns a parsed dict or None on timeout/error.

    Expected data line format:
      <solver>,<n_warmup>,<n_timed>,<sysclk_hz>,<mean>,<max>,<min>,<median>
    """
    import serial  # type: ignore

    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except serial.SerialException as exc:
        print(f"  [ERROR] Could not open {port}: {exc}")
        return None

    deadline = time.monotonic() + timeout
    data_line: str | None = None
    in_block = False

    try:
        while time.monotonic() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line == "BENCH_START":
                in_block = True
                data_line = None
            elif line == "BENCH_END" and in_block:
                break
            elif in_block and line and data_line is None:
                data_line = line
    finally:
        ser.close()

    if data_line is None:
        return None

    parts = data_line.split(",")
    if len(parts) < 8:
        return None
    try:
        return {
            "solver":        parts[0].strip(),
            "n_warmup":      int(parts[1]),
            "n_timed":       int(parts[2]),
            "sysclk_hz":     int(parts[3]),
            "cycles_mean":   int(parts[4]),
            "cycles_max":    int(parts[5]),
            "cycles_min":    int(parts[6]),
            "cycles_median": int(parts[7]),
        }
    except (ValueError, IndexError):
        return None


def _write_timings_csv(rows: list[dict], path: Path) -> None:
    """Write timing rows to a CSV compatible with command_summarize."""
    fieldnames = [
        "solver", "n_warmup", "n_timed", "sysclk_hz",
        "cycles_mean", "cycles_max", "cycles_min", "cycles_median",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_manual_flash_instructions(output_dir: Path) -> None:
    build_root = output_dir / "build"
    print(
        "\n--- Manual flash/capture instructions ---\n"
        "For each solver, flash the firmware and capture UART output:\n"
    )
    for solver in STM32_SUPPORTED_SOLVERS:
        bin_path = build_root / solver / "bench.bin"
        print(f"  {solver}:")
        print(f"    st-flash write {bin_path} 0x08000000")
        print(
            f"    # or: openocd -f board/st_nucleo_f4.cfg "
            f"-c 'program {bin_path} verify reset exit'"
        )
        print(f"    # Connect at 115200 Bd and record the BENCH_START…BENCH_END block.\n")
    print(
        "Then fill in the captured cycle counts in a CSV and run:\n"
        f"  uv run python stm32_benchmark.py summarize --output-dir {output_dir}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, build, and summarize STM32F411 solver benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical end-to-end workflow:\n"
            "  1. Generate solver code:  uv run python run_all.py\n"
            "  2. Build + run on board:  uv run python stm32_benchmark.py run\n"
            "\n"
            "Or step-by-step:\n"
            "  uv run python stm32_benchmark.py build\n"
            "  # Flash each .bin manually, capture UART output\n"
            "  uv run python stm32_benchmark.py summarize\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Shared argument groups reused across subparsers ----
    def _add_toolchain_args(p):
        p.add_argument("--cpu", default="cortex-m4",
                       help="ARM CPU name (default: cortex-m4).")
        p.add_argument("--fpu", default="fpv4-sp-d16",
                       help="FPU name (default: fpv4-sp-d16).")
        p.add_argument("--float-abi", default="hard",
                       choices=["hard", "softfp", "soft"],
                       help="Floating-point ABI (default: hard).")

    def _add_solver_args(p, defaults):
        p.add_argument("--solvers", nargs="+", default=defaults,
                       help="Solvers to process.")
        p.add_argument("--codegen-root", default=str(DEFAULT_CODEGEN_ROOT),
                       help="Root directory containing *_codegen directories.")
        p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                       help="Directory for output files.")
        p.add_argument("--problem", default="inverted_pendulum",
                       help="Problem name from problem_definition.py.")

    # ---- build ----
    build_p = subparsers.add_parser(
        "build",
        help=(
            "Generate problem_data.h and CMakeLists.txt for each solver, "
            "then build the firmware with arm-none-eabi-gcc + cmake."
        ),
    )
    _add_solver_args(build_p, STM32_SUPPORTED_SOLVERS)
    _add_toolchain_args(build_p)
    build_p.set_defaults(func=command_build)

    # ---- run ----
    run_p = subparsers.add_parser(
        "run",
        help=(
            "Build firmware, flash each solver binary to the STM32F411 board, "
            "capture DWT cycle counts over UART, and produce a summary."
        ),
    )
    _add_solver_args(run_p, STM32_SUPPORTED_SOLVERS)
    _add_toolchain_args(run_p)
    run_p.add_argument("--port", default=None,
                       help="Serial port for UART capture (e.g. /dev/ttyACM0). "
                            "Auto-detected if omitted.")
    run_p.add_argument("--flasher", default="auto",
                       choices=["auto", "st-flash", "openocd", "pyocd"],
                       help="Flash tool to use (default: auto-detect).")
    run_p.add_argument("--timeout", type=float, default=60.0,
                       help="Seconds to wait for UART output per solver (default: 60).")
    run_p.add_argument("--no-flash", action="store_true",
                       help="Build only; skip flash and capture steps.")
    run_p.set_defaults(func=command_run)

    # ---- prepare ----
    prepare = subparsers.add_parser(
        "prepare",
        help="Discover generated solver code and emit helper files for STM32 integration.",
    )
    prepare.add_argument(
        "--codegen-root",
        default=str(DEFAULT_CODEGEN_ROOT),
        help="Root directory containing *_codegen directories.",
    )
    prepare.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where manifest and helper files will be written.",
    )
    prepare.add_argument(
        "--solvers",
        nargs="+",
        default=DEFAULT_SOLVERS,
        help="Solvers to include.",
    )
    prepare.add_argument("--mcu", default="STM32F411xE", help="MCU identifier for the manifest.")
    prepare.add_argument("--cpu", default="cortex-m4", help="Target CPU passed to the ARM toolchain.")
    prepare.add_argument("--fpu", default="fpv4-sp-d16", help="Target FPU passed to the ARM toolchain.")
    prepare.add_argument(
        "--float-abi",
        default="hard",
        choices=["hard", "softfp", "soft"],
        help="Floating-point ABI passed to the ARM toolchain.",
    )
    prepare.set_defaults(func=command_prepare)

    # ---- summarize ----
    summarize = subparsers.add_parser(
        "summarize",
        help="Merge STM32 timing CSV data with arm-none-eabi-size reports.",
    )
    summarize.add_argument(
        "--manifest",
        default=str(DEFAULT_OUTPUT_DIR / "manifest.json"),
        help="Manifest produced by the prepare command.",
    )
    summarize.add_argument(
        "--timings",
        default=str(DEFAULT_OUTPUT_DIR / "timings_template.csv"),
        help="CSV file containing on-target timing measurements.",
    )
    summarize.add_argument(
        "--size-dir",
        default=str(DEFAULT_OUTPUT_DIR / "measurements"),
        help="Directory containing one size report per solver.",
    )
    summarize.add_argument(
        "--size-suffix",
        default=".size",
        help="Suffix used for size report files, e.g. lmpc.size.",
    )
    summarize.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where summary files will be written.",
    )
    summarize.set_defaults(func=command_summarize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
