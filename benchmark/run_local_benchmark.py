#!/usr/bin/env python3
"""Controlled local FastVLM INT8-vs-INT4 benchmark for Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import plistlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VARIANTS = {"int8": "8-bit", "int4": "4-bit"}
FIELDS = {
    "image_id", "variant", "caption", "TTFT_ms", "total_latency_ms",
    "generated_tokens", "tokens_per_second",
}


def die(message: str) -> None:
    raise SystemExit(message)


def command_text(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"<unavailable: {error}>"
    text = "\n".join(x.strip() for x in (result.stdout, result.stderr) if x.strip())
    return text if result.returncode == 0 else f"<exit {result.returncode}>\n{text}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def find_one(root: Path, name: str) -> Path | None:
    matches = [path for path in root.rglob(name) if path.is_dir()]
    return matches[0] if len(matches) == 1 else None


def resolve_app(path: Path) -> tuple[Path, Path]:
    app = path.expanduser().resolve()
    if not app.is_dir() or app.suffix != ".app":
        die(f"--app must point to a built macOS .app bundle: {app}")
    plist_path = app / "Contents" / "Info.plist"
    with plist_path.open("rb") as handle:
        executable_name = plistlib.load(handle).get("CFBundleExecutable")
    if not executable_name:
        die(f"CFBundleExecutable is missing from {plist_path}")
    executable = app / "Contents" / "MacOS" / executable_name
    if not executable.is_file():
        die(f"App executable was not found: {executable}")
    return app, executable


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        die(f"Could not read benchmark manifest {path}: {error}")
    if not isinstance(entries, list) or not entries:
        die("Benchmark manifest must be a non-empty JSON array.")

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            die("Every benchmark manifest entry must be an object.")
        for key in ("image_id", "image_path", "prompt"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                die(f"Manifest entry has invalid {key!r}: {entry}")
        if entry["image_id"] in seen:
            die(f"Duplicate image_id in manifest: {entry['image_id']}")
        seen.add(entry["image_id"])
        image = (path.parent / entry["image_path"]).resolve()
        if not image.is_file():
            die(f"Benchmark image was not found: {image}")
    return entries


def model_sizes(app: Path) -> dict[str, Any]:
    int8 = find_one(app, "int8.bundle")
    int4 = find_one(app, "int4.bundle")
    vision = find_one(app, "fastvithd.mlmodelc")
    if not int8 or not int4:
        return {"available": False}
    value: dict[str, Any] = {
        "available": True,
        "8-bit": {"bundle_bytes": directory_bytes(int8)},
        "4-bit": {"bundle_bytes": directory_bytes(int4)},
    }
    if vision:
        value["shared_vision"] = {"bundle_bytes": directory_bytes(vision)}
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_raw(path: Path, variant: str, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        die(f"Benchmark output was not written: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != expected_rows:
        die(f"Expected {expected_rows} {variant} records, found {len(records)}.")
    for record in records:
        if set(record) != FIELDS or record["variant"] != variant:
            die(f"Unexpected benchmark record: {record}")
        values = (
            float(record["TTFT_ms"]),
            float(record["total_latency_ms"]),
            float(record["tokens_per_second"]),
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            die(f"Invalid benchmark metrics: {record}")
        if values[1] < values[0] or int(record["generated_tokens"]) <= 0:
            die(f"Invalid benchmark metrics: {record}")
        if not str(record["caption"]).strip():
            die(f"Empty benchmark caption: {record}")
    return records


def run_process(
    executable: Path,
    manifest: Path,
    output: Path,
    log: Path,
    variant_key: str,
    warmups: int,
    max_tokens: int,
    timeout: int,
) -> float:
    env = os.environ.copy()
    env.update({
        "FASTVLM_BENCHMARK": "1",
        "FASTVLM_BENCHMARK_MANIFEST": str(manifest),
        "FASTVLM_BENCHMARK_OUTPUT": str(output),
        "FASTVLM_BENCHMARK_VARIANT": variant_key,
        "FASTVLM_BENCHMARK_WARMUP_RUNS": str(warmups),
        "FASTVLM_BENCHMARK_MAX_TOKENS": str(max_tokens),
    })
    output.unlink(missing_ok=True)
    start = time.monotonic()
    try:
        result = subprocess.run(
            [str(executable)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        partial = "\n".join(
            str(value or "") for value in (error.stdout, error.stderr)
        )
        log.write_text(partial, encoding="utf-8")
        die(f"{VARIANTS[variant_key]} process timed out. See {log}")
    combined = "\n".join(value for value in (result.stdout, result.stderr) if value)
    log.write_text(combined, encoding="utf-8")
    print(combined, end="" if combined.endswith("\n") else "\n")
    if result.returncode != 0:
        die(f"{VARIANTS[variant_key]} process exited {result.returncode}. See {log}")
    return time.monotonic() - start


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=repo / "benchmark/benchmark_manifest.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        die("Run the final benchmark natively on an Apple Silicon Mac (Darwin/arm64).")
    if args.repetitions <= 0 or args.warmups < 0 or args.max_tokens <= 0 or args.timeout_seconds <= 0:
        die("Repetitions/max-tokens/timeout must be positive; warmups must be non-negative.")

    manifest = args.manifest.expanduser().resolve()
    entries = load_manifest(manifest)
    app, executable = resolve_app(args.app)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (args.output_dir or repo / "benchmark/results" / timestamp).expanduser().resolve()
    raw_dir, log_dir = out / "raw", out / "logs"
    raw_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True)
    measurements = out / "measurements.jsonl"
    measurements.write_text("", encoding="utf-8")

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": command_text(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "app_bundle": str(app),
        "app_build_configuration": app.parent.name if app.parent.name in {"Debug", "Release"} else None,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "protocol": {
            "repetitions": args.repetitions,
            "warmup_runs_per_variant_process": args.warmups,
            "warmup_input": "first manifest entry",
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "variant_process_isolation": True,
            "variant_order": "alternates by repetition",
            "model_loading_in_timing": False,
            "ttft": "before input preparation through first generated token",
            "total_latency": "before input preparation through generation completion",
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_version": platform.mac_ver()[0],
            "sw_vers": command_text(["sw_vers"]),
            "hardware": command_text(["system_profiler", "SPHardwareDataType"]),
            "power": command_text(["pmset", "-g", "batt"]),
            "power_settings": command_text(["pmset", "-g", "custom"]),
            "xcode": command_text(["xcodebuild", "-version"]),
            "python": sys.version,
        },
        "model_sizes": model_sizes(app),
        "sessions": [],
    }
    metadata_path = out / "metadata.json"
    write_json(metadata_path, metadata)

    print(f"Results: {out}")
    try:
        for repetition in range(1, args.repetitions + 1):
            order = ["int8", "int4"] if repetition % 2 else ["int4", "int8"]
            print(f"\nRepetition {repetition}/{args.repetitions}: {' -> '.join(VARIANTS[x] for x in order)}")
            for order_index, key in enumerate(order, 1):
                stem = f"rep-{repetition:02d}-order-{order_index:02d}-{key}"
                raw, log = raw_dir / f"{stem}.jsonl", log_dir / f"{stem}.log"
                print(f"\nLaunching fresh {VARIANTS[key]} process...")
                elapsed = run_process(
                    executable, manifest, raw, log, key,
                    args.warmups, args.max_tokens, args.timeout_seconds,
                )
                records = validate_raw(raw, VARIANTS[key], len(entries))
                with measurements.open("a", encoding="utf-8") as handle:
                    for record in records:
                        record["repetition"] = repetition
                        record["order_in_repetition"] = order_index
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                metadata["sessions"].append({
                    "repetition": repetition,
                    "order_in_repetition": order_index,
                    "variant": VARIANTS[key],
                    "process_elapsed_seconds": elapsed,
                    "raw_output": str(raw),
                    "log": str(log),
                })
                write_json(metadata_path, metadata)
    except BaseException:
        metadata["status"] = "failed"
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(metadata_path, metadata)
        raise

    metadata["status"] = "completed"
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(metadata_path, metadata)
    print(f"\nMeasurements: {measurements}")
    print(f"Analyze: python3 benchmark/analyze_results.py {measurements}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
