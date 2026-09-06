#!/usr/bin/env python3
"""Aggregate FastVLM local benchmark measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VARIANTS = ("8-bit", "4-bit")
METRICS = ("TTFT_ms", "total_latency_ms", "tokens_per_second", "generated_tokens")


def die(message: str) -> None:
    raise SystemExit(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        die(f"Measurements were not found: {path}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            die(f"Invalid JSON at line {number}: {error}")
        required = {
            "image_id", "variant", "caption", "TTFT_ms", "total_latency_ms",
            "generated_tokens", "tokens_per_second", "repetition", "order_in_repetition",
        }
        if not required.issubset(record) or record["variant"] not in VARIANTS:
            die(f"Invalid record at line {number}: {record}")
        for metric in ("TTFT_ms", "total_latency_ms", "tokens_per_second"):
            value = float(record[metric])
            if not math.isfinite(value) or value <= 0:
                die(f"Invalid {metric} at line {number}: {value}")
        records.append(record)
    if not records:
        die("No measurement records were found.")
    return records


def load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def stats(records: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    values = [float(record[metric]) for record in records]
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def lower_better(int8: float, int4: float) -> float:
    return (int8 - int4) / int8 * 100.0


def higher_better(int8: float, int4: float) -> float:
    return (int4 - int8) / int8 * 100.0


def pct(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{value:.2f}%"


def size_summary(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    sizes = metadata.get("model_sizes", {})
    if not isinstance(sizes, dict) or not sizes.get("available"):
        return None
    try:
        int8 = int(sizes["8-bit"]["bundle_bytes"])
        int4 = int(sizes["4-bit"]["bundle_bytes"])
    except (KeyError, TypeError, ValueError):
        return None

    result: dict[str, Any] = {
        "8-bit_llm_bytes": int8,
        "4-bit_llm_bytes": int4,
        "8-bit_llm_mib": int8 / 2**20,
        "4-bit_llm_mib": int4 / 2**20,
        "int4_llm_storage_reduction_pct": lower_better(float(int8), float(int4)),
    }
    try:
        vision = int(sizes["shared_vision"]["bundle_bytes"])
    except (KeyError, TypeError, ValueError):
        return result

    int8_total, int4_total = int8 + vision, int4 + vision
    result.update({
        "shared_vision_bytes": vision,
        "shared_vision_mib": vision / 2**20,
        "8-bit_effective_model_mib": int8_total / 2**20,
        "4-bit_effective_model_mib": int4_total / 2**20,
        "int4_effective_model_storage_reduction_pct": lower_better(
            float(int8_total), float(int4_total)
        ),
    })
    return result


def summarize(records: list[dict[str, Any]], metadata: dict[str, Any] | None) -> dict[str, Any]:
    images = sorted({record["image_id"] for record in records})
    repetitions = sorted({int(record["repetition"]) for record in records})
    expected = {(image, repetition) for image in images for repetition in repetitions}

    by_variant = {
        variant: [record for record in records if record["variant"] == variant]
        for variant in VARIANTS
    }
    for variant, items in by_variant.items():
        counts = Counter((item["image_id"], int(item["repetition"])) for item in items)
        if set(counts) != expected or any(count != 1 for count in counts.values()):
            die(f"{variant} does not contain exactly one row per image/repetition.")

    result: dict[str, Any] = {
        "image_count": len(images),
        "repetition_count": len(repetitions),
        "record_count": len(records),
        "variants": {
            variant: {metric: stats(items, metric) for metric in METRICS}
            for variant, items in by_variant.items()
        },
    }
    int8, int4 = result["variants"]["8-bit"], result["variants"]["4-bit"]
    result["int4_vs_int8"] = {
        "median_ttft_improvement_pct": lower_better(
            int8["TTFT_ms"]["median"], int4["TTFT_ms"]["median"]
        ),
        "median_total_latency_improvement_pct": lower_better(
            int8["total_latency_ms"]["median"], int4["total_latency_ms"]["median"]
        ),
        "median_throughput_improvement_pct": higher_better(
            int8["tokens_per_second"]["median"], int4["tokens_per_second"]["median"]
        ),
    }

    sizes = size_summary(metadata)
    if sizes:
        result["model_sizes"] = sizes

    paired = {
        (record["image_id"], int(record["repetition"]), record["variant"]): record
        for record in records
    }
    matches = sum(
        paired[(image, repetition, "8-bit")]["caption"].strip()
        == paired[(image, repetition, "4-bit")]["caption"].strip()
        for image, repetition in expected
    )
    result["caption_exact_match"] = {
        "matching_pairs": matches,
        "total_pairs": len(expected),
        "rate": matches / len(expected),
        "note": "Exact string agreement is not a semantic quality score.",
    }
    return result


def per_image_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["image_id"], record["variant"])].append(record)

    rows = []
    for image in sorted({record["image_id"] for record in records}):
        a, b = grouped[(image, "8-bit")], grouped[(image, "4-bit")]
        med = lambda items, metric: statistics.median(float(x[metric]) for x in items)
        a_ttft, b_ttft = med(a, "TTFT_ms"), med(b, "TTFT_ms")
        a_total, b_total = med(a, "total_latency_ms"), med(b, "total_latency_ms")
        a_tps, b_tps = med(a, "tokens_per_second"), med(b, "tokens_per_second")
        rows.append({
            "image_id": image,
            "int8_median_TTFT_ms": a_ttft,
            "int4_median_TTFT_ms": b_ttft,
            "int4_TTFT_improvement_pct": lower_better(a_ttft, b_ttft),
            "int8_median_total_latency_ms": a_total,
            "int4_median_total_latency_ms": b_total,
            "int4_total_latency_improvement_pct": lower_better(a_total, b_total),
            "int8_median_tokens_per_second": a_tps,
            "int4_median_tokens_per_second": b_tps,
            "int4_throughput_improvement_pct": higher_better(a_tps, b_tps),
        })
    return rows


def captions_markdown(records: list[dict[str, Any]]) -> str:
    repetitions = sorted({int(record["repetition"]) for record in records})
    first = repetitions[0]
    by_key = {
        (record["image_id"], int(record["repetition"]), record["variant"]): record
        for record in records
    }
    lines = [
        "# FastVLM caption comparison", "",
        "Representative captions use the first measured repetition. Review correctness, "
        "important details, hallucinations, and coherence manually.", "",
    ]
    for image in sorted({record["image_id"] for record in records}):
        matches = sum(
            by_key[(image, rep, "8-bit")]["caption"].strip()
            == by_key[(image, rep, "4-bit")]["caption"].strip()
            for rep in repetitions
        )
        lines += [
            f"## {image}", "",
            f"Exact matches across repetitions: {matches}/{len(repetitions)}", "",
            "### 8-bit", "", by_key[(image, first, "8-bit")]["caption"].strip(), "",
            "### 4-bit", "", by_key[(image, first, "4-bit")]["caption"].strip(), "",
        ]
    return "\n".join(lines)


def summary_markdown(summary: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    a, b = summary["variants"]["8-bit"], summary["variants"]["4-bit"]
    d = summary["int4_vs_int8"]
    rows: list[tuple[str, str, str, str]] = []

    if "model_sizes" in summary:
        sizes = summary["model_sizes"]
        rows.append((
            "LLM bundle size",
            f"{sizes['8-bit_llm_mib']:.2f} MiB",
            f"{sizes['4-bit_llm_mib']:.2f} MiB",
            pct(sizes["int4_llm_storage_reduction_pct"]),
        ))
        if "shared_vision_mib" in sizes:
            rows.append((
                "Effective model footprint (LLM + shared vision)",
                f"{sizes['8-bit_effective_model_mib']:.2f} MiB",
                f"{sizes['4-bit_effective_model_mib']:.2f} MiB",
                pct(sizes["int4_effective_model_storage_reduction_pct"]),
            ))

    rows += [
        ("Median TTFT", f"{a['TTFT_ms']['median']:.2f} ms", f"{b['TTFT_ms']['median']:.2f} ms",
         pct(d["median_ttft_improvement_pct"])),
        ("Mean TTFT", f"{a['TTFT_ms']['mean']:.2f} ms", f"{b['TTFT_ms']['mean']:.2f} ms", ""),
        ("Median total latency", f"{a['total_latency_ms']['median']:.2f} ms",
         f"{b['total_latency_ms']['median']:.2f} ms", pct(d["median_total_latency_improvement_pct"])),
        ("Mean total latency", f"{a['total_latency_ms']['mean']:.2f} ms",
         f"{b['total_latency_ms']['mean']:.2f} ms", ""),
        ("Median throughput", f"{a['tokens_per_second']['median']:.2f} tok/s",
         f"{b['tokens_per_second']['median']:.2f} tok/s", pct(d["median_throughput_improvement_pct"])),
        ("Median generated tokens", f"{a['generated_tokens']['median']:.1f}",
         f"{b['generated_tokens']['median']:.1f}", ""),
    ]

    lines = [
        "# FastVLM INT8 vs INT4 local benchmark", "",
        f"Dataset: {summary['image_count']} images × {summary['repetition_count']} repetitions "
        f"per variant ({summary['record_count']} records).", "",
        "| Metric | 8-bit | 4-bit | INT4 improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {metric} | {x} | {y} | {change} |" for metric, x, y, change in rows]
    match = summary["caption_exact_match"]
    lines += [
        "",
        "Positive INT4 improvement means lower/faster for latency, higher/faster for "
        "throughput, and smaller for storage.", "",
        f"Exact caption matches: {match['matching_pairs']}/{match['total_pairs']} "
        f"({match['rate'] * 100:.1f}%). This is not a semantic quality score.",
    ]
    if metadata:
        protocol, machine = metadata.get("protocol", {}), metadata.get("machine", {})
        lines += [
            "", "## Recorded protocol", "",
            f"- Repository commit: `{metadata.get('repository_commit', '<unknown>')}`",
            f"- Build configuration: `{metadata.get('app_build_configuration', '<unknown>')}`",
            f"- Architecture: `{machine.get('machine', '<unknown>')}`",
            f"- macOS: `{machine.get('mac_version', '<unknown>')}`",
            f"- Max tokens: `{protocol.get('max_tokens', '<unknown>')}`",
            f"- Warm-ups per fresh variant process: `{protocol.get('warmup_runs_per_variant_process', '<unknown>')}`",
            f"- Variant order: `{protocol.get('variant_order', '<unknown>')}`",
        ]
    lines += [
        "", "## Interpretation guardrails", "",
        "- Model loading is excluded from TTFT and total latency.",
        "- Total latency depends partly on output length; read it with generated-token count and tokens/second.",
        "- Review `captions.md` before making a quality claim.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    measurements = args.measurements.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve() if args.metadata else measurements.parent / "metadata.json"
    output = args.output_dir.expanduser().resolve() if args.output_dir else measurements.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(measurements)
    metadata = load_metadata(metadata_path)
    summary = summarize(records, metadata)
    per_image = per_image_rows(records)

    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = summary_markdown(summary, metadata)
    (output / "summary.md").write_text(markdown)
    (output / "captions.md").write_text(captions_markdown(records))
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)

    print(markdown)
    print(f"\nAnalysis directory: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
