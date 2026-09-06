# FastVLM-Compare local benchmark

This directory contains the controlled **local Apple Silicon** measurement workflow for the FastVLM-0.5B Stage 3 INT8-vs-INT4 comparison.

GitHub Actions remains infrastructure validation only. Do not use hosted-runner timings as final project results.

## Protocol

The default local protocol is intentionally small but repeatable:

- Target: a native Apple Silicon Mac (`Darwin/arm64`).
- Model: the same FastVLM-0.5B Stage 3 checkpoint for both variants.
- Vision encoder: shared between INT8 and INT4.
- LLM quantization: 8-bit vs 4-bit, group size 64.
- Decoding: temperature `0.0`.
- Generation cap: `64` tokens.
- Dataset: five fixed repository assets in `benchmark_manifest.json`.
- Warm-up: one generation using the first manifest entry in each fresh variant process.
- Measured repetitions: five.
- Process isolation: INT8 and INT4 run in separate application processes.
- Ordering: odd repetitions run INT8 then INT4; even repetitions run INT4 then INT8.
- Model loading: excluded from TTFT and total-latency metrics.
- TTFT start: immediately before multimodal input preparation.
- TTFT end: first generated token.
- Total-latency start: same point as TTFT.
- Total-latency end: generation completion.

With five images and five repetitions, the default protocol produces **50 measured records**: 25 INT8 and 25 INT4. Warm-up generations are not written to the measurement JSONL.

### Why separate processes?

The CI smoke harness can execute both variants in one process because its job is only to prove that the native benchmark works.

For final measurements, separate processes reduce cross-variant contamination from model residency, MLX state, and caches. Alternating the order between repetitions also reduces systematic thermal/order bias.

### Before running

Use the same Mac for the entire comparison. Run natively on Apple Silicon, connect the Mac to power, disable Low Power Mode, and close heavy background workloads. Start the experiment from a reasonably idle machine and avoid changing system conditions during the run.

The local runner records macOS/hardware information, the power-source report, repository commit, manifest hash, protocol settings, and discovered model-bundle sizes in `metadata.json`.

## 1. Prepare the local app

The final benchmark must use locally generated model artifacts because CI artifacts are intentionally ephemeral.

The helper below mirrors the pinned checkpoint, dependency, export, quantization, and staging choices used by the validated workflow. It builds the local benchmark app in **Release** configuration by default:

```bash
PYTHON_BIN=python3.10 bash benchmark/prepare_local_app.sh
```

The script prints the built `.app` path when it succeeds.

Heavy intermediate files are stored under `~/Library/Caches/FastVLM-Compare/` by default. The staged model resources remain under the already-ignored `app/FastVLM/model` directory.

## 2. Run the benchmark

Use the `.app` path printed by the preparation script:

```bash
python3 benchmark/run_local_benchmark.py \
  --app "/absolute/path/to/FastVLM App.app"
```

Useful overrides:

```bash
python3 benchmark/run_local_benchmark.py \
  --app "/absolute/path/to/FastVLM App.app" \
  --repetitions 5 \
  --warmups 1 \
  --max-tokens 64
```

Do not change these settings between variants. The runner itself applies the same settings to both.

Results are written under:

```text
benchmark/results/<timestamp>/
    metadata.json
    measurements.jsonl
    raw/
    logs/
```

Each raw process output uses the existing Swift JSONL schema. `measurements.jsonl` adds only:

- `repetition`
- `order_in_repetition`

These annotations let the analysis pair equivalent INT8/INT4 runs without changing the CI smoke-test schema.

## 3. Analyze results

Run:

```bash
python3 benchmark/analyze_results.py \
  benchmark/results/<timestamp>/measurements.jsonl
```

This creates:

```text
analysis/
    summary.json
    summary.md
    per_image.csv
    captions.md
```

The summary reports:

- median and mean TTFT
- median and mean total latency
- median and mean tokens/second
- median generated-token count
- INT4 relative improvement for TTFT, total latency, and throughput
- INT4 LLM-bundle storage reduction and effective LLM-plus-shared-vision reduction when sizes were discovered
- exact caption agreement as a reproducibility signal

`captions.md` intentionally presents paired captions for manual review. Exact string agreement is **not** treated as a semantic quality metric.

## Dataset

`benchmark_manifest.json` is a self-contained practical suite using assets already versioned in the repository. It covers a chart/diagram, a flexible-prompt visual, counting, emoji/symbol interpretation, and handwriting/text reading.

Animated GIF assets are decoded deterministically from frame 0 by the Swift benchmark runner.

This is not intended to replace a large academic multimodal benchmark. Its purpose is to provide a fixed, reproducible set of distinct visual tasks for the engineering comparison.

The existing `benchmark_manifest.example.json` remains the one-image CI smoke manifest.

## Interpreting the metrics

A lower TTFT is better. A lower total latency is generally better, but total latency is also affected by how many tokens each variant generated, so read it together with generated-token count and tokens/second.

A higher tokens/second value is better for decoder throughput.

Model loading is intentionally excluded from TTFT and total latency. This keeps the experiment focused on input preparation plus inference/generation. If cold-start/model-load performance becomes a project goal later, measure it as a separate metric rather than mixing it into these values.

For output quality, inspect whether each caption preserves the correct objects/text/counts, important details, coherence, and whether one variant introduces hallucinations. Do not claim that one variant has better semantic quality from exact-string matching alone.
