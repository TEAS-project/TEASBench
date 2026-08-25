#!/usr/bin/env python3
"""Validate and stage required EIDF SWE-bench evidence as exact LFS pointers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED_LFS_URL = "https://gitlab.eidf.ac.uk/teas/lfs.git/info/lfs"
DATASET = "swe-bench-lite"
POINTER_VERSION = "https://git-lfs.github.com/spec/v1"


class EvidenceError(ValueError):
    """Raised when a run cannot support a trustworthy publication."""


def _run(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise EvidenceError(f"git {' '.join(args)} failed: {detail}")
    return proc


def _single(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise EvidenceError(
            f"expected exactly one {label} matching {pattern!r}, found {len(matches)}"
        )
    path = matches[0]
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise EvidenceError(f"{label} must be one non-empty regular file: {path}")
    return path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise EvidenceError(f"blank JSONL line {line_number}: {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"invalid JSONL line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise EvidenceError(f"non-object JSONL line {line_number}: {path}")
            rows.append(row)
    if not rows:
        raise EvidenceError(f"empty JSONL evidence: {path}")
    return rows


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object: {path}")
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise EvidenceError(f"cannot derive a path slug from {value!r}")
    return slug


def model_slug(model_name: str) -> str:
    return _slug(model_name.rstrip("/").rsplit("/", 1)[-1])


def hardware_slug(gpu_type: str, num_gpus: int) -> str:
    tokens = gpu_type.split()
    return f"{_slug(tokens[-1] if tokens else gpu_type)}x{num_gpus}"


def validate_source(
    source: Path,
    *,
    expected_tasks: int,
    call_limit: int,
    task_timeout_s: int,
    expected_engine: str,
    expected_engine_version: str,
    expected_model_name: str,
    expected_gpu_type: str,
    expected_num_gpus: int,
    concurrency: int,
) -> list[Path]:
    source = source.resolve()
    if not source.is_dir():
        raise EvidenceError(f"source run directory does not exist: {source}")

    results_path = _single(source, "results.jsonl", "results evidence")
    detailed_path = _single(
        source, f"detailed-results_{DATASET}_*.jsonl", "detailed request evidence"
    )
    output_path = _single(
        source, f"output-data_{DATASET}_*.jsonl", "per-task output evidence"
    )

    detailed_suffix = detailed_path.name.removeprefix("detailed-results_").removesuffix(".jsonl")
    output_suffix = output_path.name.removeprefix("output-data_").removesuffix(".jsonl")
    if detailed_suffix != output_suffix:
        raise EvidenceError(
            f"detailed/output run identity differs: {detailed_suffix!r} != {output_suffix!r}"
        )
    metadata_path = _single(source, f"metadata_{detailed_suffix}.json", "metadata")
    metrics_path = _single(source, f"metrics_{detailed_suffix}.json", "metrics")

    results = load_jsonl(results_path)
    detailed = load_jsonl(detailed_path)
    outputs = load_jsonl(output_path)
    if len(results) != expected_tasks or len(outputs) != expected_tasks:
        raise EvidenceError(
            f"expected {expected_tasks} task rows, found results={len(results)} "
            f"output-data={len(outputs)}"
        )

    result_ids = [row.get("task_id") for row in results]
    output_ids = [row.get("task_id") for row in outputs]
    if any(not isinstance(task_id, str) or not task_id for task_id in result_ids):
        raise EvidenceError("every results row must have a non-empty string task_id")
    if len(set(result_ids)) != len(result_ids):
        raise EvidenceError("results.jsonl contains duplicate task_id rows")
    if any(not isinstance(task_id, str) or not task_id for task_id in output_ids):
        raise EvidenceError("every output-data row must have a non-empty string task_id")
    if len(set(output_ids)) != len(output_ids) or set(output_ids) != set(result_ids):
        raise EvidenceError("output-data task identity does not match results.jsonl")
    results_by_id = dict(zip(result_ids, results))

    task_dirs = sorted(path for path in source.glob("task_*") if path.is_dir())
    task_dir_ids = [path.name.removeprefix("task_") for path in task_dirs]
    if len(task_dir_ids) != expected_tasks or set(task_dir_ids) != set(result_ids):
        raise EvidenceError("runtime task directories do not identify the exact result population")
    detailed_index_by_task = {
        task_id: index for index, task_id in enumerate(task_dir_ids)
    }

    expected_indices = list(range(expected_tasks))
    output_indices = [row.get("index") for row in outputs]
    if output_indices != expected_indices:
        raise EvidenceError("output-data indices must be the exact contiguous task population")
    for index, output in enumerate(outputs):
        result = results_by_id[output["task_id"]]
        usage = result.get("total_usage")
        if not isinstance(usage, dict):
            raise EvidenceError(f"results[{index}].total_usage is missing")
        expected_output_usage = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "num_requests": usage.get("requests"),
        }
        mismatches = {
            key: (output.get(key), value)
            for key, value in expected_output_usage.items()
            if output.get(key) != value
        }
        if mismatches:
            raise EvidenceError(
                f"results/output-data usage mismatch for task {output['task_id']}: {mismatches}"
            )

    request_counts: Counter[int] = Counter()
    request_indices: dict[int, list[int]] = {}
    token_totals: dict[int, list[int]] = {
        index: [0, 0] for index in expected_indices
    }
    for row in detailed:
        example_index = row.get("example_index")
        request_index = row.get("request_index")
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        if (
            not isinstance(example_index, int)
            or isinstance(example_index, bool)
            or not 0 <= example_index < expected_tasks
            or not isinstance(request_index, int)
            or isinstance(request_index, bool)
            or request_index < 0
        ):
            raise EvidenceError("detailed request indices are invalid or outside the run")
        if (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 0
            or not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
        ):
            raise EvidenceError("detailed token counts must be non-negative integers")
        if input_tokens == 0 and output_tokens == 0:
            raise EvidenceError(
                f"detailed request {example_index}/{request_index} has both token counts zero"
            )
        request_counts[example_index] += 1
        request_indices.setdefault(example_index, []).append(request_index)
        token_totals[example_index][0] += input_tokens
        token_totals[example_index][1] += output_tokens

    for example_index, indices in request_indices.items():
        if sorted(indices) != list(range(len(indices))):
            raise EvidenceError(
                f"detailed request indices are not contiguous for example {example_index}"
            )
    for index, row in enumerate(outputs):
        detailed_index = detailed_index_by_task[row["task_id"]]
        num_requests = row.get("num_requests")
        if num_requests != request_counts[detailed_index]:
            raise EvidenceError(
                f"output-data[{index}].num_requests={num_requests!r}, "
                f"detailed evidence has {request_counts[detailed_index]}"
            )
        if row.get("input_tokens") != token_totals[detailed_index][0]:
            raise EvidenceError(f"output-data[{index}] input tokens do not match detailed evidence")
        if row.get("output_tokens") != token_totals[detailed_index][1]:
            raise EvidenceError(f"output-data[{index}] output tokens do not match detailed evidence")

    metadata = _load_json(metadata_path, "metadata")
    metrics = _load_json(metrics_path, "metrics")
    environment = metadata.get("system_environment")
    if not isinstance(environment, dict):
        raise EvidenceError("metadata.system_environment is missing")
    hardware = metadata.get("hardware")
    if not isinstance(hardware, dict):
        raise EvidenceError("metadata.hardware is missing")
    model = metadata.get("model_config")
    if not isinstance(model, dict):
        raise EvidenceError("metadata.model_config is missing")
    expected_policy = {
        "dataset": DATASET,
        "num_examples": expected_tasks,
        "sweagent_call_limit": call_limit,
        "sweagent_task_timeout_s": task_timeout_s,
        "inference_engine": expected_engine,
        "inference_engine_version": expected_engine_version,
        "tensor_parallel_size": expected_num_gpus,
        "concurrency": concurrency,
        "observed_max_concurrency": concurrency,
    }
    mismatches = {
        key: (environment.get(key), value)
        for key, value in expected_policy.items()
        if environment.get(key) != value
    }
    if mismatches:
        raise EvidenceError(f"metadata execution policy mismatch: {mismatches}")
    expected_hardware = {"gpu_type": expected_gpu_type, "num_gpus": expected_num_gpus}
    hardware_mismatches = {
        key: (hardware.get(key), value)
        for key, value in expected_hardware.items()
        if hardware.get(key) != value
    }
    if hardware_mismatches:
        raise EvidenceError(f"metadata hardware mismatch: {hardware_mismatches}")
    if model.get("model_name") != expected_model_name:
        raise EvidenceError(
            f"metadata model_name {model.get('model_name')!r} != {expected_model_name!r}"
        )
    metrics_hardware = metrics.get("hardware")
    expected_metrics_hardware = {
        "gpu_type": expected_gpu_type,
        "num_gpus": expected_num_gpus,
        f"{expected_engine}_version": expected_engine_version,
    }
    if not isinstance(metrics_hardware, dict) or any(
        metrics_hardware.get(key) != value
        for key, value in expected_metrics_hardware.items()
    ):
        raise EvidenceError("metrics hardware identity does not match expected run")
    quality = metrics.get("quality")
    if not isinstance(quality, dict) or quality.get("total_examples") != expected_tasks:
        raise EvidenceError("metrics quality denominator does not match expected task population")

    return [results_path, detailed_path, output_path]


def validate_destination(
    relative: str,
    *,
    expected_engine: str,
    expected_model_name: str,
    expected_gpu_type: str,
    expected_num_gpus: int,
) -> PurePosixPath:
    path = PurePosixPath(relative)
    parts = path.parts
    if path.is_absolute() or ".." in parts or len(parts) != 8:
        raise EvidenceError(f"invalid publication destination: {relative!r}")
    if (
        parts[0:2] != ("agentic", "eidf")
        or parts[2] != expected_engine
        or parts[3] != model_slug(expected_model_name)
        or parts[4] != DATASET
        or parts[5] != hardware_slug(expected_gpu_type, expected_num_gpus)
        or not parts[6].startswith("batch-size-")
        or not re.fullmatch(r"[0-9]{8}[-_][0-9]{4,6}", parts[7])
    ):
        raise EvidenceError(f"destination is not an exact EIDF SWE run leaf: {relative!r}")
    return path


def validate_repo_policy(repo: Path, relative: PurePosixPath, names: list[str]) -> None:
    repo = repo.resolve()
    top = _run(repo, "rev-parse", "--show-toplevel").stdout.strip()
    if Path(top).resolve() != repo:
        raise EvidenceError(f"--repo must name the git top level: {repo}")
    config_path = repo / ".lfsconfig"
    if not config_path.is_file():
        raise EvidenceError(f"missing tracked .lfsconfig: {config_path}")
    _run(repo, "ls-files", "--error-unmatch", ".lfsconfig")
    configured = _run(
        repo, "config", "--file", str(config_path), "--get", "lfs.url"
    ).stdout.strip()
    if configured != EXPECTED_LFS_URL:
        raise EvidenceError(
            f"unexpected LFS endpoint {configured!r}; expected {EXPECTED_LFS_URL!r}"
        )
    local_override = _run(repo, "config", "--local", "--get", "lfs.url", check=False)
    if local_override.returncode == 0 and local_override.stdout.strip() != EXPECTED_LFS_URL:
        raise EvidenceError(f"local lfs.url overrides the expected endpoint: {local_override.stdout.strip()!r}")

    for name in names:
        rel_file = (relative / name).as_posix()
        output = _run(repo, "check-attr", "filter", "diff", "merge", "--", rel_file).stdout
        attributes = {}
        for line in output.splitlines():
            _path, attribute, value = line.rsplit(": ", 2)
            attributes[attribute] = value
        if attributes != {"filter": "lfs", "diff": "lfs", "merge": "lfs"}:
            raise EvidenceError(f"mandatory evidence is not LFS-tracked: {rel_file}: {attributes}")


def parse_pointer(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    if len(lines) != 3 or lines[0] != f"version {POINTER_VERSION}":
        raise EvidenceError("staged blob is not an exact Git LFS pointer")
    oid = re.fullmatch(r"oid sha256:([0-9a-f]{64})", lines[1])
    size = re.fullmatch(r"size ([0-9]+)", lines[2])
    if oid is None or size is None:
        raise EvidenceError("staged blob has an invalid Git LFS oid or size")
    return oid.group(1), int(size.group(1))


def stage(repo: Path, relative: PurePosixPath, sources: list[Path]) -> list[str]:
    repo = repo.resolve()
    destination = repo.joinpath(*relative.parts)
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    if repo not in destination_resolved.parents:
        raise EvidenceError(f"destination escapes repository: {destination}")
    for source in sources:
        target = destination / source.name
        if target.exists() or target.is_symlink():
            raise EvidenceError(f"refusing to overwrite existing evidence: {target}")

    copied: list[Path] = []
    try:
        for source in sources:
            target = destination / source.name
            shutil.copyfile(source, target)
            copied.append(target)
        _run(repo, "lfs", "install", "--local")
        rel_files = [target.relative_to(repo).as_posix() for target in copied]
        _run(repo, "add", "-f", "--", *rel_files)
        for source, rel_file in zip(sources, rel_files):
            pointer_text = _run(repo, "show", f":{rel_file}").stdout
            oid, size = parse_pointer(pointer_text)
            source_bytes = source.read_bytes()
            expected_oid = hashlib.sha256(source_bytes).hexdigest()
            if oid != expected_oid or size != len(source_bytes):
                raise EvidenceError(
                    f"staged LFS pointer does not match immutable source bytes: {rel_file}"
                )
        return rel_files
    except Exception:
        for target in copied:
            target.unlink(missing_ok=True)
        raise


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or str(parsed) != value:
        raise argparse.ArgumentTypeError("expected a canonical positive integer")
    return parsed


def nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("expected a non-empty value")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--expected-tasks", required=True, type=positive_int)
    parser.add_argument("--sweagent-call-limit", required=True, type=positive_int)
    parser.add_argument("--sweagent-task-timeout-s", required=True, type=positive_int)
    parser.add_argument("--expected-engine", required=True, choices=("sglang", "vllm"))
    parser.add_argument("--expected-engine-version", required=True, type=nonempty)
    parser.add_argument("--expected-model-name", required=True, type=nonempty)
    parser.add_argument("--expected-gpu-type", required=True, type=nonempty)
    parser.add_argument("--expected-num-gpus", required=True, type=positive_int)
    parser.add_argument("--concurrency", required=True, type=positive_int)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--destination-relative")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source, destination and LFS policy without copying or staging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = validate_source(
            args.source_run_dir,
            expected_tasks=args.expected_tasks,
            call_limit=args.sweagent_call_limit,
            task_timeout_s=args.sweagent_task_timeout_s,
            expected_engine=args.expected_engine,
            expected_engine_version=args.expected_engine_version,
            expected_model_name=args.expected_model_name,
            expected_gpu_type=args.expected_gpu_type,
            expected_num_gpus=args.expected_num_gpus,
            concurrency=args.concurrency,
        )
        if args.repo is None or args.destination_relative is None:
            raise EvidenceError("--repo and --destination-relative are required")
        relative = validate_destination(
            args.destination_relative,
            expected_engine=args.expected_engine,
            expected_model_name=args.expected_model_name,
            expected_gpu_type=args.expected_gpu_type,
            expected_num_gpus=args.expected_num_gpus,
        )
        validate_repo_policy(args.repo, relative, [path.name for path in sources])
        if args.dry_run:
            report = {
                "mode": "dry-run",
                "destination": relative.as_posix(),
                "lfs_url": EXPECTED_LFS_URL,
                "source_files": [path.name for path in sources],
            }
        else:
            staged = stage(args.repo, relative, sources)
            report = {"mode": "stage", "staged": staged, "lfs_url": EXPECTED_LFS_URL}
        print(json.dumps(report, sort_keys=True))
        return 0
    except (EvidenceError, OSError, subprocess.SubprocessError) as exc:
        print(f"agentic evidence staging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
