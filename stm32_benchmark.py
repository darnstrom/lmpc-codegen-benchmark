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
import re
import shlex
import stat
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and summarize STM32F411 solver benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
