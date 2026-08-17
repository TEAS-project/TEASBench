#!/usr/bin/env python3
"""Validate controlled-variation launch evidence and immutable state."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import sys
import tempfile


STUDY_ID = "controlled-variation-2026-x2"
PREFLIGHT_KIND = "excluded-compatibility-preflight"
PREFLIGHT_COMBINATIONS = {
    ("vllm", "0.16.0"),
    ("vllm", "0.21.0"),
    ("sglang", "0.5.9"),
    ("sglang", "0.5.12.post1"),
}
IMAGE_BASES = {
    "vllm": "vllm/vllm-openai",
    "sglang": "lmsysorg/sglang",
}
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{7,40}\Z")
FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
JOB_UID_RE = re.compile(r"[0-9a-f][0-9a-f-]{7,}\Z")


class GuardError(ValueError):
    pass


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise GuardError(f"manifest not found: {path}")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GuardError(f"{path}: line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise GuardError(f"{path}: line {line_number}: expected a JSON object")
        records.append(record)
    if not records:
        raise GuardError(f"manifest has no records: {path}")
    return records


def read_pins(path):
    path = Path(path)
    if not path.is_file():
        raise GuardError(f"image-pin file not found: {path}")
    pins = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not DIGEST_RE.fullmatch(fields[1]):
            raise GuardError(f"{path}: line {line_number}: expected TAG<TAB>sha256:DIGEST")
        tag, digest = fields
        if tag in pins:
            raise GuardError(f"{path}: duplicate image tag {tag}")
        pins[tag] = digest
    if not pins:
        raise GuardError(f"image-pin file has no records: {path}")
    return pins


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, text, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def command_block_complete(args):
    complete = set()
    nodes = set()
    for record in read_jsonl(args.manifest):
        if record.get("study_id") != STUDY_ID or record.get("block") != args.block:
            raise GuardError(
                f"{args.manifest}: record does not belong to {STUDY_ID}/{args.block}")
        order = record.get("planned_order")
        if not isinstance(order, int) or order not in range(1, 13):
            raise GuardError(f"{args.manifest}: invalid planned_order {order!r}")
        if record.get("outcome") == "complete":
            receipt_path = Path(str(record.get("receipt_path", "")))
            receipt_hash = record.get("receipt_sha256")
            if (not receipt_path.is_file() or not isinstance(receipt_hash, str)
                    or not HEX_RE.fullmatch(receipt_hash)
                    or sha256(receipt_path) != receipt_hash):
                raise GuardError(
                    f"{args.manifest}: complete record {order} has no valid receipt")
            receipt = load_json(receipt_path, "receipt")
            yaml_path = Path(str(record.get("yaml_path", "")))
            yaml_hash = record.get("yaml_sha256")
            if (not yaml_path.is_file() or not isinstance(yaml_hash, str)
                    or not HEX_RE.fullmatch(yaml_hash)
                    or sha256(yaml_path) != yaml_hash
                    or receipt.get("job_yaml_sha256") != yaml_hash):
                raise GuardError(
                    f"{args.manifest}: complete record {order} has no frozen submitted YAML")
            output_path = validate_receipt_data(
                receipt, block=args.block, order=order,
                job=record.get("job"), job_uid=record.get("job_uid"),
                node=record.get("node"), image_id=record.get("image_id"),
                teasbench_commit=record.get("teasbench_commit"),
                moe_cap_ref=record.get("moe_cap_ref"),
                context=f"{args.manifest}: receipt for order {order}")
            require_equal(record, "output_path", output_path,
                          f"{args.manifest}: complete record {order}")
            complete.add(order)
        node = record.get("node")
        if node is not None and not isinstance(node, str):
            raise GuardError(f"{args.manifest}: node must be a string")
        if node:
            nodes.add(node)
    if len(nodes) > 1:
        raise GuardError(
            f"{args.block} manifest records multiple nodes: {', '.join(sorted(nodes))}")
    missing = sorted(set(range(1, 13)) - complete)
    if missing:
        raise GuardError(
            f"{args.block} is not complete; missing successful leaves "
            + ",".join(map(str, missing)))


def command_manifest_node(args):
    nodes = []
    for record in read_jsonl(args.manifest):
        if record.get("study_id") != STUDY_ID or record.get("block") != args.block:
            raise GuardError(
                f"{args.manifest}: record does not belong to {STUDY_ID}/{args.block}")
        node = record.get("node")
        if node is not None and not isinstance(node, str):
            raise GuardError(f"{args.manifest}: node must be a string")
        if node:
            nodes.append(node)
    distinct = sorted(set(nodes))
    if len(distinct) > 1:
        raise GuardError(
            f"{args.block} manifest records multiple nodes: {', '.join(distinct)}")
    print(nodes[-1] if nodes else "")


def command_validate_csv(args):
    pipeline_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(pipeline_dir))
    from utils import STUDY_COORDINATES, study_fields

    with Path(args.csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 72:
        raise GuardError(f"study CSV must contain exactly 72 rows, got {len(rows)}")
    seen = set()
    for line_number, row in enumerate(rows, 2):
        try:
            block, _ = study_fields(row)
        except ValueError as exc:
            raise GuardError(f"{args.csv}: line {line_number}: {exc}") from exc
        order = int(row["study_order"], 10)
        key = (block, order)
        if key in seen:
            raise GuardError(f"{args.csv}: duplicate frozen coordinate {block}/{order}")
        seen.add(key)
    expected = {(block, order) for block, leaves in STUDY_COORDINATES.items()
                for order in leaves}
    if seen != expected:
        raise GuardError("study CSV does not contain the complete frozen coordinate set")


def command_write_state(args):
    if not args.value:
        raise GuardError("state value must be non-empty")
    atomic_write(args.path, args.value.rstrip("\n") + "\n")


def command_append_manifest(args):
    require_commit(args.teasbench_commit, "--teasbench-commit", full=True)
    require_commit(args.moe_cap_ref, "--moe-cap-ref", full=True)
    record = {
        "study_id": STUDY_ID,
        "block": args.block,
        "planned_order": args.order,
        "job": args.job,
        "job_uid": args.job_uid,
        "yaml": args.yaml,
        "yaml_path": args.yaml_path,
        "yaml_sha256": args.yaml_sha256,
        "node": args.node,
        "image_id": args.image_id,
        "moe_cap_ref": args.moe_cap_ref,
        "teasbench_commit": args.teasbench_commit,
        "submitted_at": args.submitted_at,
        "finished_at": args.finished_at,
        "outcome": args.outcome,
        "receipt_path": args.receipt_path,
        "receipt_sha256": args.receipt_sha256,
        "output_path": args.output_path,
    }
    if args.outcome == "submitted":
        for key in ("job", "job_uid", "yaml", "yaml_path", "yaml_sha256",
                    "submitted_at"):
            if not record[key]:
                raise GuardError(f"submitted manifest record requires {key}")
        if args.finished_at:
            raise GuardError("submitted manifest record cannot have finished_at")
    elif not args.finished_at:
        raise GuardError(f"{args.outcome} manifest record requires finished_at")
    if args.outcome == "complete":
        for key in ("node", "image_id", "receipt_path", "receipt_sha256", "output_path"):
            if not record[key]:
                raise GuardError(f"complete manifest record requires {key}")
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_prepare_job(args):
    source = Path(args.yaml)
    if not source.is_file():
        raise GuardError(f"generated YAML not found: {source}")
    text = source.read_text()
    pattern = re.compile(r"^(  generateName: )([a-z0-9.-]+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise GuardError(f"{source}: expected exactly one metadata.generateName")
    prefix = matches[0].group(2)
    suffix = "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                     for _ in range(5))
    job = prefix + suffix
    if len(job) > 63 or not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", job):
        raise GuardError(f"{source}: generated invalid Kubernetes job name {job!r}")
    prepared = pattern.sub(f"  name: {job}", text, count=1)
    destination = Path(args.state_dir) / f"{job}.yaml"
    atomic_write(destination, prepared, mode=source.stat().st_mode)
    print(f"{job}\t{destination.resolve()}\t{sha256(destination)}")


def image_digest(value, context):
    if not isinstance(value, str) or "@" not in value:
        raise GuardError(f"{context} must be digest-qualified")
    digest = value.rsplit("@", 1)[1]
    if not DIGEST_RE.fullmatch(digest):
        raise GuardError(f"{context} must end in a sha256 digest")
    return digest


def expected_study_coordinate(block, order):
    pipeline_dir = Path(__file__).resolve().parents[2]
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    from utils import STUDY_COORDINATES
    try:
        return STUDY_COORDINATES[block.lower()][order]
    except (KeyError, AttributeError) as exc:
        raise GuardError(f"invalid frozen study coordinate {block}/{order}") from exc


def validate_receipt_data(receipt, *, block, order, job, job_uid, node, image_id,
                          teasbench_commit, moe_cap_ref, context):
    require_commit(teasbench_commit, f"{context} expected teasbench_commit", full=True)
    require_commit(moe_cap_ref, f"{context} expected moe_cap_ref", full=True)
    expected_engine, expected_version, expected_dataset = expected_study_coordinate(
        block, order)
    expected = {
        "receipt_version": 1,
        "status": "validated",
        "study_id": STUDY_ID,
        "block": block.lower(),
        "planned_order": order,
        "job": job,
        "job_uid": job_uid,
        "node": node,
        "publication": "development",
        "inference_engine": expected_engine,
        "engine_version": expected_version,
        "dataset": expected_dataset,
    }
    for key, expected_value in expected.items():
        require_equal(receipt, key, expected_value, context)
    for key in ("metadata_sha256", "metrics_sha256", "job_yaml_sha256"):
        if not isinstance(receipt.get(key), str) or not HEX_RE.fullmatch(receipt[key]):
            raise GuardError(f"{context}: {key} must be 64 lowercase hex digits")
    receipt_image_digest = image_digest(receipt.get("image_ref"), f"{context}: image_ref")
    if image_id is not None and image_digest(image_id, f"{context}: image_id") != receipt_image_digest:
        raise GuardError(f"{context}: runtime image digest does not match the receipt")
    artifact_hashes = receipt.get("artifact_sha256")
    expected_artifacts = {
        "metadata", "metrics", "launch_yaml", "detailed_results",
        "output_data", "timings", "pip_freeze",
    }
    if expected_engine == "sglang":
        expected_artifacts.add("expert_distribution_bundle")
    if (not isinstance(artifact_hashes, dict)
            or set(artifact_hashes) != expected_artifacts
            or any(not isinstance(value, str) or not HEX_RE.fullmatch(value)
                   for value in artifact_hashes.values())):
        raise GuardError(f"{context}: artifact_sha256 is incomplete or invalid")
    for receipt_key, artifact_key in (
            ("metadata_sha256", "metadata"), ("metrics_sha256", "metrics"),
            ("job_yaml_sha256", "launch_yaml")):
        if receipt[receipt_key] != artifact_hashes[artifact_key]:
            raise GuardError(
                f"{context}: {receipt_key} disagrees with artifact_sha256")
    receipt_teas = require_commit(receipt.get("teasbench_commit"),
                                  f"{context} teasbench_commit")
    receipt_moe = require_commit(receipt.get("moe_cap_commit"),
                                 f"{context} moe_cap_commit")
    if not teasbench_commit.startswith(receipt_teas):
        raise GuardError(f"{context}: receipt TEASBench commit does not match")
    if not moe_cap_ref.startswith(receipt_moe):
        raise GuardError(f"{context}: receipt MoE-CAP commit does not match")
    output_path = receipt.get("output_path")
    publish_path = receipt.get("publish_path")
    if not isinstance(output_path, str) or not Path(output_path).is_absolute():
        raise GuardError(f"{context}: output_path must be absolute")
    if f"study-{block.lower()}" not in Path(output_path).parts:
        raise GuardError(f"{context}: output_path is outside the expected study block")
    if not isinstance(publish_path, str) or not publish_path:
        raise GuardError(f"{context}: publish_path must be non-empty")
    if not output_path.rstrip("/").endswith("/" + publish_path.strip("/")):
        raise GuardError(f"{context}: output_path and publish_path do not identify one run")
    quality = receipt.get("quality")
    if quality != {"total": 256, "attempted": 256, "served": 256, "completed": 256}:
        raise GuardError(f"{context}: receipt quality counts are not all 256")
    return output_path


def command_validate_receipt(args):
    require_commit(args.teasbench_commit, "--teasbench-commit", full=True)
    require_commit(args.moe_cap_ref, "--moe-cap-ref", full=True)
    receipt = load_json(args.receipt, "receipt")
    output_path = validate_receipt_data(
        receipt, block=args.block, order=args.order, job=args.job,
        job_uid=args.job_uid, node=args.node, image_id=args.image_id,
        teasbench_commit=args.teasbench_commit, moe_cap_ref=args.moe_cap_ref,
        context=str(args.receipt))
    job_yaml = Path(args.job_yaml)
    if not job_yaml.is_file() or sha256(job_yaml) != receipt["job_yaml_sha256"]:
        raise GuardError(f"{args.receipt}: receipt does not hash-bind the submitted YAML")
    try:
        import yaml
    except ImportError as exc:
        raise GuardError("PyYAML is required to validate the submitted YAML") from exc
    try:
        document = yaml.safe_load(job_yaml.read_text())
        containers = document["spec"]["template"]["spec"]["containers"]
        if not isinstance(containers, list) or len(containers) != 1:
            raise GuardError(f"{args.receipt}: submitted YAML must have one container")
        submitted_image = containers[0]["image"]
    except GuardError:
        raise
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise GuardError(f"{args.receipt}: cannot parse submitted YAML: {exc}") from exc
    if submitted_image != receipt["image_ref"]:
        raise GuardError(f"{args.receipt}: receipt image differs from submitted YAML")
    print(f"{sha256(args.receipt)}\t{output_path}")


def command_repeat_action(args):
    latest = None
    for record in read_jsonl(args.manifest):
        if record.get("study_id") != STUDY_ID or record.get("block") != args.block:
            raise GuardError(
                f"{args.manifest}: record does not belong to {STUDY_ID}/{args.block}")
        if record.get("planned_order") == args.order:
            latest = record
    if latest is None:
        if args.reconcile:
            raise GuardError(
                f"{args.block}/{args.order} has no ambiguous identity to reconcile")
        print("new")
        return
    outcome = latest.get("outcome")
    ambiguous_identity = outcome in {
        "identity-unknown", "create-failed", "config-copy-failed",
        "failed", "timeout", "missing-job",
    }
    if args.reconcile and not ambiguous_identity:
        raise GuardError(
            f"{args.block}/{args.order} has no ambiguous identity to reconcile")
    if outcome == "complete":
        raise GuardError(f"{args.block}/{args.order} is already scientifically complete")
    if outcome == "submitted":
        for key in ("job", "job_uid", "yaml", "yaml_path", "yaml_sha256",
                    "submitted_at"):
            if not isinstance(latest.get(key), str) or not latest[key]:
                raise GuardError(
                    f"{args.block}/{args.order} submitted record has no {key}")
        yaml_path = Path(latest["yaml_path"])
        if (not yaml_path.is_file()
                or not isinstance(latest["yaml_sha256"], str)
                or not HEX_RE.fullmatch(latest["yaml_sha256"])
                or sha256(yaml_path) != latest["yaml_sha256"]):
            raise GuardError(
                f"{args.block}/{args.order} submitted YAML is missing or changed")
        print("\t".join(("resume", latest["job"], latest["job_uid"],
                         latest["yaml"], latest["submitted_at"],
                         str(yaml_path), latest["yaml_sha256"])))
        return
    if ambiguous_identity:
        for key in ("job", "yaml", "yaml_path", "yaml_sha256", "submitted_at"):
            if not isinstance(latest.get(key), str) or not latest[key]:
                raise GuardError(
                    f"{args.block}/{args.order} ambiguous identity record has no {key}")
        yaml_path = Path(latest["yaml_path"])
        if (not yaml_path.is_file()
                or not HEX_RE.fullmatch(latest["yaml_sha256"])
                or sha256(yaml_path) != latest["yaml_sha256"]):
            raise GuardError(
                f"{args.block}/{args.order} ambiguous identity YAML is missing or changed")
        if not args.reconcile:
            raise GuardError(
                f"{args.block}/{args.order} has an unreconciled named-job identity; "
                "automatic resubmission is prohibited")
        prior_uid = str(latest.get("job_uid", ""))
        if not JOB_UID_RE.fullmatch(prior_uid):
            prior_uid = "unresolved"
        print("\t".join(("reconcile", latest["job"], prior_uid, latest["yaml"],
                         latest["submitted_at"], str(yaml_path),
                         latest["yaml_sha256"])))
        return
    print("new")


def command_pin_images(args):
    resolved = read_pins(args.resolved_file)
    pin_path = Path(args.pin_file)
    existing = read_pins(pin_path) if pin_path.exists() else {}
    for tag, digest in resolved.items():
        old_digest = existing.get(tag)
        if old_digest and old_digest != digest:
            raise GuardError(
                f"image tag drift for {tag}: frozen {old_digest}, registry now {digest}")

    rendered = {}
    image_line = re.compile(r"^(\s*image:\s*)(\S+)(\s*)$", re.MULTILINE)
    immutable = {tag: f"{tag.rsplit(':', 1)[0]}@{digest}"
                 for tag, digest in resolved.items()}
    for yaml_path in map(Path, args.yaml):
        text = yaml_path.read_text()
        replacements = 0

        def replace(match):
            nonlocal replacements
            tag = match.group(2)
            if tag not in immutable:
                return match.group(0)
            replacements += 1
            return f"{match.group(1)}{immutable[tag]}{match.group(3)}"

        updated = image_line.sub(replace, text)
        if replacements != 1:
            raise GuardError(
                f"{yaml_path}: expected exactly one resolved serving image, got {replacements}")
        rendered[yaml_path] = updated

    for yaml_path, text in rendered.items():
        atomic_write(yaml_path, text, mode=yaml_path.stat().st_mode)

    if args.persist:
        merged = {**existing, **resolved}
        pin_text = "".join(f"{tag}\t{merged[tag]}\n" for tag in sorted(merged))
        atomic_write(pin_path, pin_text)

    for tag in sorted(resolved):
        print(f"{tag} -> {immutable[tag]}")


def load_json(path, label):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GuardError(f"{label} {path} must contain a JSON object")
    return value


def require_equal(record, key, expected, context):
    actual = record.get(key)
    if actual != expected:
        raise GuardError(f"{context}: {key} must be {expected!r}, got {actual!r}")


def require_commit(value, context, full=False):
    pattern = FULL_COMMIT_RE if full else COMMIT_RE
    if not isinstance(value, str) or not pattern.fullmatch(value):
        kind = "40 lowercase hex digits" if full else "7-40 lowercase hex digits"
        raise GuardError(f"{context} must be {kind}")
    return value


def require_inside(path, root, label):
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise GuardError(f"cannot resolve {label} {path}: {exc}") from exc
    if not resolved.is_file():
        raise GuardError(f"{label} is not a regular file: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GuardError(f"{label} resolves outside artifact_dir: {resolved}") from exc
    return resolved


def flag_value(tokens, flag, context):
    matches = [index for index, token in enumerate(tokens) if token == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(tokens):
        raise GuardError(f"{context}: expected exactly one {flag} value")
    return tokens[matches[0] + 1]


def command_tokens(script, module, context):
    logical = script.replace("\\\n", " ")
    pattern = re.compile(
        rf"^[ \t]*(?:python3|python)\s+-m\s+{re.escape(module)}\s+"
        rf"(.+?)(?:&>|2>|>)",
        re.DOTALL | re.MULTILINE)
    match = pattern.search(logical)
    if not match:
        raise GuardError(f"{context}: command for {module} not found")
    try:
        return ["python", "-m", module, *shlex.split(match.group(1))]
    except ValueError as exc:
        raise GuardError(f"{context}: cannot parse {module} command: {exc}") from exc


def command_validate_preflight(args):
    require_commit(args.teasbench_commit, "--teasbench-commit", full=True)
    require_commit(args.moe_cap_ref, "--moe-cap-ref", full=True)
    pins = read_pins(args.image_pins)
    records = read_jsonl(args.manifest)
    if len(records) != 4:
        raise GuardError(f"preflight manifest must contain exactly 4 records, got {len(records)}")

    seen = set()
    artifact_dirs = set()
    job_uids = set()
    for index, record in enumerate(records, 1):
        context = f"{args.manifest}: record {index}"
        require_equal(record, "study_id", STUDY_ID, context)
        require_equal(record, "kind", PREFLIGHT_KIND, context)
        require_equal(record, "gpu", "A100", context)
        require_equal(record, "num_gpu", 2, context)
        require_equal(record, "dataset", "longbench_v1", context)
        require_equal(record, "num_samples", 256, context)
        require_equal(record, "batch_size", "default", context)
        require_equal(record, "outcome", "complete", context)
        require_equal(record, "teasbench_commit", args.teasbench_commit, context)
        require_equal(record, "moe_cap_ref", args.moe_cap_ref, context)

        engine = record.get("inference_engine")
        version = record.get("engine_version")
        combination = (engine, version)
        if combination not in PREFLIGHT_COMBINATIONS:
            raise GuardError(f"{context}: unexpected engine/build {combination!r}")
        if combination in seen:
            raise GuardError(f"{context}: duplicate engine/build {combination!r}")
        seen.add(combination)

        image_tag = f"{IMAGE_BASES[engine]}:v{version}"
        require_equal(record, "image_tag", image_tag, context)
        if image_tag not in pins:
            raise GuardError(f"{context}: {image_tag} is absent from the frozen image pins")
        image_ref = f"{IMAGE_BASES[engine]}@{pins[image_tag]}"
        require_equal(record, "image_ref", image_ref, context)

        for key in ("job_uid", "job_name", "node"):
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                raise GuardError(f"{context}: {key} must be a non-empty string")
        if record["job_uid"] in job_uids:
            raise GuardError(f"{context}: duplicate job_uid {record['job_uid']!r}")
        job_uids.add(record["job_uid"])
        gpu_uuids = record.get("gpu_uuids")
        if (not isinstance(gpu_uuids, list) or len(gpu_uuids) != 2
                or any(not isinstance(item, str) or not item for item in gpu_uuids)
                or len(set(gpu_uuids)) != 2):
            raise GuardError(f"{context}: gpu_uuids must contain two distinct values")

        completed_at = record.get("completed_at")
        if not isinstance(completed_at, str):
            raise GuardError(f"{context}: completed_at must be an ISO-8601 string")
        try:
            completed = dt.datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise GuardError(f"{context}: invalid completed_at {completed_at!r}") from exc
        if completed.tzinfo is None:
            raise GuardError(f"{context}: completed_at must include a timezone")
        if completed > dt.datetime.now(dt.timezone.utc):
            raise GuardError(f"{context}: completed_at is in the future")

        artifact_dir = Path(str(record.get("artifact_dir", "")))
        if not artifact_dir.is_absolute():
            raise GuardError(f"{context}: artifact_dir must be an existing absolute directory")
        try:
            resolved_artifact = artifact_dir.resolve(strict=True)
        except OSError as exc:
            raise GuardError(f"{context}: cannot resolve artifact_dir: {exc}") from exc
        if not resolved_artifact.is_dir():
            raise GuardError(f"{context}: artifact_dir is not a directory")
        if any(re.fullmatch(r"study-e[1-6]", part) for part in resolved_artifact.parts):
            raise GuardError(f"{context}: preflight artifact is inside a study block path")
        if "compatibility-preflight" not in resolved_artifact.parts:
            raise GuardError(
                f"{context}: artifact_dir must have a compatibility-preflight path segment")
        if resolved_artifact in artifact_dirs:
            raise GuardError(f"{context}: duplicate artifact_dir {resolved_artifact}")
        artifact_dirs.add(resolved_artifact)

        metadata_path = require_inside(
            resolved_artifact / "metadata.json", resolved_artifact, "metadata.json")
        metrics_path = require_inside(
            resolved_artifact / "metrics.json", resolved_artifact, "metrics.json")
        job_yamls = sorted((*resolved_artifact.glob("*.yaml"),
                            *resolved_artifact.glob("*.yml")))
        if len(job_yamls) != 1:
            raise GuardError(
                f"{context}: artifact_dir must contain exactly one job YAML, got {len(job_yamls)}")
        job_yaml_path = require_inside(job_yamls[0], resolved_artifact, "job YAML")
        metadata_hash = record.get("metadata_sha256")
        metrics_hash = record.get("metrics_sha256")
        job_yaml_hash = record.get("job_yaml_sha256")
        if not isinstance(metadata_hash, str) or not HEX_RE.fullmatch(metadata_hash):
            raise GuardError(f"{context}: metadata_sha256 must be 64 lowercase hex digits")
        if not isinstance(metrics_hash, str) or not HEX_RE.fullmatch(metrics_hash):
            raise GuardError(f"{context}: metrics_sha256 must be 64 lowercase hex digits")
        if not isinstance(job_yaml_hash, str) or not HEX_RE.fullmatch(job_yaml_hash):
            raise GuardError(f"{context}: job_yaml_sha256 must be 64 lowercase hex digits")
        if sha256(metadata_path) != metadata_hash:
            raise GuardError(f"{context}: metadata.json hash does not match the manifest")
        if sha256(metrics_path) != metrics_hash:
            raise GuardError(f"{context}: metrics.json hash does not match the manifest")
        if sha256(job_yaml_path) != job_yaml_hash:
            raise GuardError(f"{context}: job YAML hash does not match the manifest")
        try:
            import yaml
        except ImportError as exc:
            raise GuardError("PyYAML is required to validate the preflight job") from exc
        try:
            job_document = yaml.safe_load(job_yaml_path.read_text())
            pod_spec = job_document["spec"]["template"]["spec"]
            containers = pod_spec["containers"]
            if not isinstance(containers, list) or len(containers) != 1:
                raise GuardError(f"{context}: job YAML must have exactly one container")
            container = containers[0]
            require_equal(container, "image", image_ref, f"{context} job YAML container")
            require_equal(container.get("resources", {}).get("limits", {}),
                          "nvidia.com/gpu", 2, f"{context} job YAML limits")
            require_equal(pod_spec.get("nodeSelector", {}), "nvidia.com/gpu.product",
                          "NVIDIA-A100-SXM4-80GB", f"{context} job YAML nodeSelector")
            args_list = container.get("args")
            if not isinstance(args_list, list) or len(args_list) != 1:
                raise GuardError(f"{context}: job YAML container must have one script argument")
            job_script = args_list[0]
            if not isinstance(job_script, str):
                raise GuardError(f"{context}: job YAML script must be a string")
        except GuardError:
            raise
        except (KeyError, TypeError, yaml.YAMLError) as exc:
            raise GuardError(f"{context}: invalid Kubernetes job YAML: {exc}") from exc

        server_module = f"moe_cap.systems.{engine}"
        server_tokens = command_tokens(job_script, server_module, context)
        model_flag = "--model" if engine == "vllm" else "--model-path"
        require_equal({model_flag: flag_value(server_tokens, model_flag, context)},
                      model_flag, "unsloth/gpt-oss-120b", context)
        tp_flag = "--tensor-parallel-size" if engine == "vllm" else "--tp-size"
        require_equal({tp_flag: flag_value(server_tokens, tp_flag, context)},
                      tp_flag, "2", context)
        batch_flag = "--max-num-seqs" if engine == "vllm" else "--max-running-requests"
        if batch_flag in server_tokens:
            raise GuardError(
                f"{context}: default batch recipe must not set {batch_flag}")
        # These tails are the exact semantic commands rendered by the frozen
        # E1 A100x2 LongBench study coordinates. The excluded preflight may
        # change only its output location/provenance wrapper, not serving or
        # client flags.
        if engine == "vllm":
            expected_server_tail = [
                "--model", "unsloth/gpt-oss-120b",
                "--port", "30000", "--host", "0.0.0.0",
                "--tensor-parallel-size", "2",
                "--reasoning-parser", "openai_gptoss",
            ]
        else:
            expected_server_tail = [
                "--model-path", "unsloth/gpt-oss-120b",
                "--port", "30000",
                "--expert-distribution-recorder-mode", "stat",
                "--tp-size", "2", "--reasoning-parser", "gpt-oss",
            ]
        if server_tokens[3:] != expected_server_tail:
            raise GuardError(
                f"{context}: server command differs from the frozen A100x2 recipe")
        client_tokens = command_tokens(
            job_script, "moe_cap.runner.openai_api_profile", context)
        expected_client = {
            "--model_name": "unsloth/gpt-oss-120b",
            "--datasets": "longbench_v1",
            "--num-samples": "256",
        }
        for flag, expected_value in expected_client.items():
            require_equal({flag: flag_value(client_tokens, flag, context)},
                          flag, expected_value, context)
        if "--server-batch-size" in client_tokens:
            raise GuardError(
                f"{context}: default batch recipe must not set --server-batch-size")
        output_dir = flag_value(client_tokens, "--output_dir", context)
        output_parts = Path(output_dir).parts
        if ("batch-size-default" not in output_parts
                or "compatibility-preflight" not in output_parts):
            raise GuardError(
                f"{context}: client output is not the excluded default-batch preflight")
        expected_client_tail = [
            "--model_name", "unsloth/gpt-oss-120b",
            "--datasets", "longbench_v1", "--num-samples", "256",
            "--api-url", "http://localhost:30000/v1/completions",
            "--backend", engine, "--output_dir", output_dir, "--use-chat-api",
        ]
        if client_tokens[3:] != expected_client_tail:
            raise GuardError(
                f"{context}: client command differs from the frozen LongBench recipe")
        checkout_pattern = re.compile(
            rf"^[ \t]*git checkout --quiet --detach {args.moe_cap_ref}(?:[ \t]|$)",
            re.MULTILINE)
        if not checkout_pattern.search(job_script):
            raise GuardError(f"{context}: job YAML does not pin the full MoE-CAP commit")
        teas_matches = re.findall(
            r'"teasbench_commit"\s*:\s*"([0-9a-f]{7,40})"', job_script)
        if len(set(teas_matches)) != 1 or not args.teasbench_commit.startswith(teas_matches[0]):
            raise GuardError(f"{context}: job YAML TEASBench commit does not match")

        metadata = load_json(metadata_path, "metadata")
        environment = metadata.get("system_environment", {})
        hardware = metadata.get("hardware", {})
        model_config = metadata.get("model_config", {})
        preflight_metadata = metadata.get("compatibility_preflight", {})
        require_equal(model_config, "model_name", "unsloth/gpt-oss-120b",
                      f"{context} metadata")
        require_equal(environment, "inference_engine", engine, f"{context} metadata")
        require_equal(
            environment, "inference_engine_version", version, f"{context} metadata")
        metadata_teas = require_commit(
            environment.get("teasbench_commit"), f"{context} metadata teasbench_commit")
        if not args.teasbench_commit.startswith(metadata_teas):
            raise GuardError(f"{context} metadata: teasbench_commit does not match")
        metadata_moe_ref = environment.get("moe_cap_commit")
        require_commit(metadata_moe_ref, f"{context} metadata moe_cap_commit")
        if not args.moe_cap_ref.startswith(metadata_moe_ref):
            raise GuardError(
                f"{context} metadata: moe_cap_commit does not match {args.moe_cap_ref}")
        require_equal(hardware, "num_gpus", 2, f"{context} metadata")
        require_equal(hardware, "gpu_type", "NVIDIA-A100-SXM4-80GB",
                      f"{context} metadata")
        metadata_expected = {
            "dataset": "longbench_v1", "num_samples": 256,
            "batch_size": "default", "gpu": "A100", "num_gpu": 2,
            "gpu_uuids": gpu_uuids, "job_uid": record["job_uid"],
            "job_name": record["job_name"], "node": record["node"],
            "image_ref": image_ref,
        }
        for key, expected_value in metadata_expected.items():
            require_equal(preflight_metadata, key, expected_value,
                          f"{context} metadata compatibility_preflight")

        metrics = load_json(metrics_path, "metrics")
        quality = metrics.get("quality", {})
        for key in ("total", "attempted", "served", "completed"):
            require_equal(quality, key, 256, f"{context} metrics.quality")

    if seen != PREFLIGHT_COMBINATIONS:
        missing = sorted(PREFLIGHT_COMBINATIONS - seen)
        raise GuardError(f"preflight manifest is missing engine/build combinations: {missing}")

    if args.record:
        validation = {
            "study_id": STUDY_ID,
            "kind": "validated-compatibility-preflight",
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256(args.manifest),
            "teasbench_commit": args.teasbench_commit,
            "moe_cap_ref": args.moe_cap_ref,
            "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "combinations": [list(item) for item in sorted(seen)],
        }
        atomic_write(args.record, json.dumps(validation, sort_keys=True, indent=2) + "\n")


def command_capture_preflight(args):
    record = load_json(args.receipt, "preflight receipt")
    context = str(args.receipt)
    require_equal(record, "study_id", STUDY_ID, context)
    require_equal(record, "kind", PREFLIGHT_KIND, context)
    require_equal(record, "outcome", "complete", context)
    require_equal(record, "job_name", args.job, context)
    require_equal(record, "job_uid", args.job_uid, context)
    source_artifact_dir = record.get("artifact_dir")
    if (not isinstance(source_artifact_dir, str)
            or not Path(source_artifact_dir).is_absolute()):
        raise GuardError(f"{context}: artifact_dir must be an absolute pod path")
    if args.manifest or args.artifact_dir:
        if not args.manifest or not args.artifact_dir:
            raise GuardError("--manifest and --artifact-dir must be supplied together")
        local_artifact_dir = Path(args.artifact_dir).resolve(strict=True)
        if not local_artifact_dir.is_dir():
            raise GuardError("captured preflight artifact_dir is not a directory")
        record["artifact_dir"] = str(local_artifact_dir)
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(source_artifact_dir)


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    complete = subparsers.add_parser("block-complete")
    complete.add_argument("--manifest", required=True)
    complete.add_argument("--block", required=True, choices=[f"E{i}" for i in range(1, 7)])
    complete.set_defaults(func=command_block_complete)

    node = subparsers.add_parser("manifest-node")
    node.add_argument("--manifest", required=True)
    node.add_argument("--block", required=True, choices=[f"E{i}" for i in range(1, 7)])
    node.set_defaults(func=command_manifest_node)

    matrix = subparsers.add_parser("validate-csv")
    matrix.add_argument("--csv", required=True)
    matrix.set_defaults(func=command_validate_csv)

    state = subparsers.add_parser("write-state")
    state.add_argument("--path", required=True)
    state.add_argument("--value", required=True)
    state.set_defaults(func=command_write_state)

    append = subparsers.add_parser("append-manifest")
    append.add_argument("--manifest", required=True)
    append.add_argument("--block", required=True, choices=[f"E{i}" for i in range(1, 7)])
    append.add_argument("--order", required=True, type=int, choices=range(1, 13))
    append.add_argument("--job", default="")
    append.add_argument("--job-uid", default="")
    append.add_argument("--yaml", required=True)
    append.add_argument("--yaml-path", default="")
    append.add_argument("--yaml-sha256", default="")
    append.add_argument("--node", default="")
    append.add_argument("--image-id", default="")
    append.add_argument("--moe-cap-ref", required=True)
    append.add_argument("--teasbench-commit", required=True)
    append.add_argument("--submitted-at", default="")
    append.add_argument("--finished-at", default="")
    append.add_argument(
        "--outcome", required=True,
        choices=("submitted", "create-failed", "config-copy-failed", "failed",
                 "timeout", "missing-job", "identity-unknown",
                 "identity-reconciled-absent",
                 "cleanup-confirmed",
                 "validation-failed", "complete"))
    append.add_argument("--receipt-path", default="")
    append.add_argument("--receipt-sha256", default="")
    append.add_argument("--output-path", default="")
    append.set_defaults(func=command_append_manifest)

    prepare = subparsers.add_parser("prepare-job")
    prepare.add_argument("--yaml", required=True)
    prepare.add_argument("--state-dir", required=True)
    prepare.set_defaults(func=command_prepare_job)

    repeat = subparsers.add_parser("repeat-action")
    repeat.add_argument("--manifest", required=True)
    repeat.add_argument("--block", required=True, choices=[f"E{i}" for i in range(1, 7)])
    repeat.add_argument("--order", required=True, type=int, choices=range(1, 13))
    repeat.add_argument("--reconcile", action="store_true")
    repeat.set_defaults(func=command_repeat_action)

    receipt = subparsers.add_parser("validate-receipt")
    receipt.add_argument("--receipt", required=True)
    receipt.add_argument("--block", required=True, choices=[f"E{i}" for i in range(1, 7)])
    receipt.add_argument("--order", required=True, type=int, choices=range(1, 13))
    receipt.add_argument("--job", required=True)
    receipt.add_argument("--job-uid", required=True)
    receipt.add_argument("--node", required=True)
    receipt.add_argument("--image-id", required=True)
    receipt.add_argument("--job-yaml", required=True)
    receipt.add_argument("--teasbench-commit", required=True)
    receipt.add_argument("--moe-cap-ref", required=True)
    receipt.set_defaults(func=command_validate_receipt)

    pins = subparsers.add_parser("pin-images")
    pins.add_argument("--pin-file", required=True)
    pins.add_argument("--resolved-file", required=True)
    pins.add_argument("--persist", action="store_true")
    pins.add_argument("yaml", nargs="+")
    pins.set_defaults(func=command_pin_images)

    preflight = subparsers.add_parser(
        "validate-preflight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Validate the excluded A100x2 compatibility-preflight evidence.",
        epilog="""Each of the four JSONL records must contain:
  study_id, kind, inference_engine, engine_version, gpu, num_gpu,
  dataset, num_samples, batch_size, outcome, teasbench_commit, moe_cap_ref,
  image_tag, image_ref, job_uid, job_name, node, gpu_uuids, completed_at,
  artifact_dir, metadata_sha256, metrics_sha256, job_yaml_sha256.

artifact_dir must be absolute, have a compatibility-preflight path segment,
remain outside study-e1..study-e6, and contain metadata.json, metrics.json, and
exactly one digest-pinned A100x2 job YAML. Metrics must report all 256 LongBench
samples as total, attempted, served, and completed.""")
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--image-pins", required=True)
    preflight.add_argument("--teasbench-commit", required=True)
    preflight.add_argument("--moe-cap-ref", required=True)
    preflight.add_argument("--record")
    preflight.set_defaults(func=command_validate_preflight)

    capture = subparsers.add_parser(
        "capture-preflight", description="Inspect or append one pod preflight receipt.")
    capture.add_argument("--receipt", required=True)
    capture.add_argument("--job", required=True)
    capture.add_argument("--job-uid", required=True)
    capture.add_argument("--manifest")
    capture.add_argument("--artifact-dir")
    capture.set_defaults(func=command_capture_preflight)

    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (GuardError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
