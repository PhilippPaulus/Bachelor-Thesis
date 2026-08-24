from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.cli import add_common_db_model_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preflight and all five thesis experiments in order")
    add_common_db_model_args(parser)
    parser.add_argument("--complete-workload-path", required=True)
    parser.add_argument("--include-exact-runtime-reference", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    common = _common_args(args)
    _run("run_preflight.py", [*common, "--complete-workload-path", args.complete_workload_path])
    subsequent = _subsequent_run_mode(common, overwrite=args.overwrite)
    _run("experiment_1_accuracy.py", [*subsequent, "--complete-workload-path", args.complete_workload_path])
    _run("experiment_2_failure_analysis.py", subsequent)
    _run("experiment_3_base_join_influence.py", [*subsequent, "--complete-workload-path", args.complete_workload_path])
    _run("experiment_4_base_join_quality.py", [*subsequent, "--complete-workload-path", args.complete_workload_path])
    runtime = [*subsequent, "--complete-workload-path", args.complete_workload_path]
    if args.include_exact_runtime_reference:
        runtime.append("--include-exact-reference")
    _run("experiment_5_runtime_impact.py", runtime)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("generate_evaluation_plots.py")),
            "--run-dir",
            str(Path(args.output_root).expanduser().resolve() / args.run_id),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def _run(script_name: str, args: list[str]) -> None:
    command = [sys.executable, str(Path(__file__).with_name(script_name)), *args]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _common_args(args: argparse.Namespace) -> list[str]:
    output = [
        "--model-dir", args.model_dir,
        "--output-root", args.output_root,
        "--run-id", args.run_id,
        "--schema", args.schema,
        "--random-seed", str(args.random_seed),
        "--bootstrap-seed", str(args.bootstrap_seed),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--timeout-seconds", str(args.timeout_seconds),
        "--warmups", str(args.warmups),
        "--repetitions", str(args.repetitions),
        "--cache-policy", args.cache_policy,
        "--log-level", args.log_level,
    ]
    output.extend(["--connection-file", args.connection_file] if args.connection_file else ["--conn-string", args.conn_string])
    if args.template_map:
        output.extend(["--template-map", args.template_map])
    if args.query_limit is not None:
        output.extend(["--query-limit", str(args.query_limit)])
    for query_id in args.query_id:
        output.extend(["--query-id", query_id])
    if args.sample_count is not None:
        output.extend(["--sample-count", str(args.sample_count)])
    output.append("--injection-validation" if args.injection_validation else "--no-injection-validation")
    if args.overwrite:
        output.append("--overwrite")
    elif args.resume:
        output.append("--resume")
    return output


def _subsequent_run_mode(args: list[str], *, overwrite: bool) -> list[str]:
    mode = "--overwrite" if overwrite else "--resume"
    return [value for value in args if value not in {"--overwrite", "--resume"}] + [mode]


if __name__ == "__main__":
    main()
